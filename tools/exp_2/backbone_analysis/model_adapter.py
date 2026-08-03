from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import torch

from .common import existing_file


def nested_get(value: Any, dotted_key: str | None) -> Any:
    if not dotted_key:
        return value
    current = value
    for key in dotted_key.split('.'):
        if not isinstance(current, Mapping) or key not in current:
            raise KeyError(f'Checkpoint does not contain state key: {dotted_key}')
        current = current[key]
    return current


def state_dict_from_checkpoint(checkpoint: Any, state_key: str | None) -> Mapping[str, Any]:
    selected = nested_get(checkpoint, state_key)
    if state_key:
        value = selected
    elif isinstance(selected, Mapping) and isinstance(selected.get('state_dict'), Mapping):
        value = selected['state_dict']
    elif isinstance(selected, Mapping) and isinstance(selected.get('model'), Mapping):
        value = selected['model']
    else:
        value = selected
    if not isinstance(value, Mapping):
        raise TypeError('Selected checkpoint state is not a mapping')
    tensors = {str(key): tensor for key, tensor in value.items() if torch.is_tensor(tensor)}
    if not tensors:
        raise ValueError('Selected checkpoint state contains no tensors')
    return tensors


def strip_prefix(state: Mapping[str, Any], prefix: str) -> Dict[str, Any]:
    return {
        key[len(prefix):] if key.startswith(prefix) else key: value
        for key, value in state.items()
    }


def select_backbone_state(
    state: Mapping[str, Any],
    target: Mapping[str, torch.Tensor],
    explicit_prefix: str | None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    prefixes = []
    if explicit_prefix is not None:
        prefixes.append(explicit_prefix)
    prefixes.extend([
        '', 'backbone.', 'module.', 'module.backbone.',
        'student.', 'student.backbone.', 'student.module.backbone.',
        'teacher.', 'teacher.backbone.', 'teacher.module.backbone.',
    ])
    candidates = []
    seen = set()
    for prefix in prefixes:
        if prefix in seen:
            continue
        seen.add(prefix)
        transformed = strip_prefix(state, prefix)
        matched = {
            key: value for key, value in transformed.items()
            if key in target and tuple(value.shape) == tuple(target[key].shape)
        }
        candidates.append((len(matched), prefix, matched, len(transformed)))
    candidates.sort(key=lambda item: item[0], reverse=True)
    count, prefix, matched, source_count = candidates[0]
    if count == 0:
        examples = list(state)[:10]
        raise RuntimeError(
            'No checkpoint tensors match the backbone. Example source keys: '
            f'{examples}')
    report = {
        'selected_prefix': prefix,
        'matched_tensors': count,
        'target_tensors': len(target),
        'source_tensors': source_count,
        'match_ratio': count / float(max(len(target), 1)),
    }
    return matched, report


@dataclass
class LoadedModel:
    model_id: str
    model: Any
    backbone: Any
    layers: Dict[str, Any]
    load_report: Dict[str, Any]

    def close(self) -> None:
        del self.layers
        del self.backbone
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def resolve_module(root: Any, dotted_name: str) -> Any:
    current = root
    for field in dotted_name.split('.'):
        if field.isdigit() and hasattr(current, '__getitem__'):
            current = current[int(field)]
        else:
            if not hasattr(current, field):
                raise AttributeError(
                    f'Module {type(current).__name__} has no child {field} '
                    f'while resolving {dotted_name}')
            current = getattr(current, field)
    return current


def load_model(spec: Mapping[str, Any], device: str) -> LoadedModel:
    from mmengine.config import Config
    from mmengine.runner.checkpoint import _load_checkpoint
    from mmdet.apis import init_detector

    model_id = str(spec['id'])
    kind = str(spec.get('kind', 'detector'))
    if kind not in {'backbone', 'detector'}:
        raise ValueError(f'{model_id}: kind must be backbone or detector')
    config_path = existing_file(spec['config'])
    checkpoint_path = existing_file(spec['checkpoint'])
    config = Config.fromfile(str(config_path))
    if config.model.get('backbone', {}).get('init_cfg') is not None:
        config.model.backbone.init_cfg = None

    if kind == 'detector':
        model = init_detector(config, str(checkpoint_path), device=device)
        load_report = {
            'kind': kind,
            'config': str(config_path),
            'checkpoint': str(checkpoint_path),
            'loader': 'mmdet.apis.init_detector',
        }
    else:
        model = init_detector(config, checkpoint=None, device=device)
        checkpoint = _load_checkpoint(str(checkpoint_path), map_location='cpu')
        state = state_dict_from_checkpoint(checkpoint, spec.get('state_dict_key'))
        matched, matching_report = select_backbone_state(
            state, model.backbone.state_dict(), spec.get('checkpoint_prefix'))
        result = model.backbone.load_state_dict(matched, strict=False)
        minimum_ratio = float(spec.get('minimum_match_ratio', 0.5))
        if matching_report['match_ratio'] < minimum_ratio:
            raise RuntimeError(
                f'{model_id}: checkpoint match ratio '
                f'{matching_report["match_ratio"]:.3f} is below {minimum_ratio:.3f}')
        load_report = {
            'kind': kind,
            'config': str(config_path),
            'checkpoint': str(checkpoint_path),
            'state_dict_key': spec.get('state_dict_key'),
            **matching_report,
            'missing_keys': list(result.missing_keys),
            'unexpected_keys': list(result.unexpected_keys),
        }
        del checkpoint, state, matched

    model.eval()
    layer_map = spec.get('layers') or {
        'res2': 'layer1', 'res3': 'layer2',
        'res4': 'layer3', 'res5': 'layer4',
    }
    if not isinstance(layer_map, Mapping) or not layer_map:
        raise ValueError(f'{model_id}: layers must be a non-empty mapping')
    layers = {
        str(layer_id): resolve_module(model.backbone, str(module_name))
        for layer_id, module_name in layer_map.items()
    }
    load_report['layers'] = {str(k): str(v) for k, v in layer_map.items()}
    return LoadedModel(model_id, model, model.backbone, layers, load_report)
