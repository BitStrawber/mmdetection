#!/usr/bin/env bash
set -euo pipefail

# Check which Exp2 underwater synthesis outputs are complete.
# Expected default split sizes: train=250000, val=10000.
# Models checked by default: uwnr_ruod_ref syreanet_synthesis cut_ruod watergan_ruod uwdf.

SYN_ROOT="${SYN_ROOT:-/media/HDD1/XCX/exp_2/synthetic_imagenet}"
SOURCE_ROOT="${SOURCE_ROOT:-/media/SSD1/XCX/exp_2/synthetic_imagenet}"
DEPTH_ROOT="${DEPTH_ROOT:-/media/SSD1/XCX/exp_2/depthanything_v2_maps}"
WORK_ROOT="${WORK_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work}"
MODELS="${MODELS:-uwnr_ruod_ref syreanet_synthesis cut_ruod watergan_ruod uwdf}"
SPLITS="${SPLITS:-train val}"
TRAIN_EXPECTED="${TRAIN_EXPECTED:-250000}"
VAL_EXPECTED="${VAL_EXPECTED:-10000}"
OUT_DIR="${OUT_DIR:-logs/synthesis_completion}"

mkdir -p "${OUT_DIR}"

SYN_ROOT="${SYN_ROOT}" \
SOURCE_ROOT="${SOURCE_ROOT}" \
DEPTH_ROOT="${DEPTH_ROOT}" \
WORK_ROOT="${WORK_ROOT}" \
MODELS="${MODELS}" \
SPLITS="${SPLITS}" \
TRAIN_EXPECTED="${TRAIN_EXPECTED}" \
VAL_EXPECTED="${VAL_EXPECTED}" \
OUT_DIR="${OUT_DIR}" \
python - <<'PY'
from pathlib import Path
import csv
import json
import os
from datetime import datetime

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPEG"}

syn_root = Path(os.environ["SYN_ROOT"])
source_root = Path(os.environ["SOURCE_ROOT"])
depth_root = Path(os.environ["DEPTH_ROOT"])
work_root = Path(os.environ["WORK_ROOT"])
models = os.environ["MODELS"].split()
splits = os.environ["SPLITS"].split()
out_dir = Path(os.environ["OUT_DIR"])
expected = {
    "train": int(os.environ["TRAIN_EXPECTED"]),
    "val": int(os.environ["VAL_EXPECTED"]),
}

configs = {
    "uwnr_ruod_ref": {
        "source_base": source_root / "uwnr" / "source",
        "depth_base": depth_root / "uwnr",
        "generated": {
            "train": [
                syn_root / "uwnr_ruod_ref" / "generated" / "train",
                work_root / "uwnr_ruod_ref" / "generated" / "train",
            ],
            "val": [
                syn_root / "uwnr_ruod_ref" / "generated" / "val",
                work_root / "uwnr_ruod_ref" / "generated" / "val",
            ],
        },
    },
    "syreanet_synthesis": {
        "source_base": source_root / "syreanet" / "source",
        "depth_base": depth_root / "syreanet",
        "generated": {
            "train": [
                syn_root / "syreanet_synthesis" / "generated" / "train",
                syn_root / "syreanet" / "generated" / "train",
                work_root / "syreanet_synthesis" / "generated" / "train",
            ],
            "val": [
                syn_root / "syreanet_synthesis" / "generated" / "val",
                syn_root / "syreanet" / "generated" / "val",
                work_root / "syreanet_synthesis" / "generated" / "val",
            ],
        },
    },
    "cut_ruod": {
        "source_base": source_root / "cut" / "source",
        "depth_base": depth_root / "cut",
        "generated": {
            "train": [
                syn_root / "cut" / "generated" / "train",
                work_root / "cut" / "generated" / "train",
                work_root / "cut" / "results" / "imagenet_ruod_cut_full_ssd_train",
            ],
            "val": [
                syn_root / "cut" / "generated" / "val",
                work_root / "cut" / "generated" / "val",
                work_root / "cut" / "results" / "imagenet_ruod_cut_full_ssd_val",
            ],
        },
    },
    "watergan_ruod": {
        "source_base": source_root / "watergan" / "source",
        "depth_base": depth_root / "watergan",
        "generated": {
            "train": [
                syn_root / "watergan" / "generated_step1564_official_mat" / "train",
                syn_root / "watergan" / "generated" / "train",
                work_root / "watergan" / "generated" / "train",
                work_root / "watergan" / "results" / "train",
            ],
            "val": [
                syn_root / "watergan" / "generated_step1564_official_mat" / "val",
                syn_root / "watergan" / "generated" / "val",
                work_root / "watergan" / "generated" / "val",
                work_root / "watergan" / "results" / "val",
            ],
        },
        "extra": [
            work_root / "watergan" / "datasets",
            work_root / "watergan" / "checkpoints",
            Path("/home/fcp/xcx/exp_2/syn/WaterGAN"),
        ],
    },
    "uwdf": {
        "source_base": source_root / "uwdf" / "source",
        "depth_base": depth_root / "uwdf",
        "generated": {
            "train": [
                syn_root / "uwdf" / "generated" / "train",
                work_root / "uwdf_controlnet_ipadapter" / "train",
                work_root / "uwdf_ipadapter" / "train",
            ],
            "val": [
                syn_root / "uwdf" / "generated" / "val",
                work_root / "uwdf_controlnet_ipadapter" / "val",
                work_root / "uwdf_ipadapter" / "val",
            ],
        },
        "debug_roots": [
            work_root / "uwdf_condition_linkage_seven_ablation" / "experiments",
            work_root / "uwdf_blur_ref_strength_sweep" / "experiments",
            work_root / "uwdf_blur_ref_high_strength_sweep" / "experiments",
            work_root / "uwdf_depth_ablation" / "experiments",
        ],
    },
}

def count_images(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for x in path.rglob("*") if x.is_file() and x.suffix in IMG_EXTS)

def newest_meta(path: Path, limit=3):
    if not path.exists():
        return []
    files = []
    for pattern in ("summary.json", "manifest.jsonl", "*.log"):
        files.extend(x for x in path.rglob(pattern) if x.is_file())
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return files[:limit]

def best_generated(paths):
    rows = [(count_images(p), str(p), p.exists()) for p in paths]
    rows.sort(reverse=True, key=lambda x: x[0])
    return rows[0], rows

def status(count, target):
    if count >= target:
        return "DONE"
    if count == 0:
        return "MISSING"
    return "PARTIAL"

rows = []
unfinished = []
print("============================================================")
print("Exp2 synthesis completion check")
print(f"SYN_ROOT:    {syn_root}")
print(f"SOURCE_ROOT: {source_root}")
print(f"DEPTH_ROOT:  {depth_root}")
print(f"WORK_ROOT:   {work_root}")
print("Expected: train={train}, val={val}".format(**expected))
print("============================================================")

for model in models:
    cfg = configs.get(model)
    if cfg is None:
        print(f"\n===== {model} =====")
        print("UNKNOWN MODEL CONFIG")
        continue

    print(f"\n===== {model} =====")
    model_done = True
    for split in splits:
        target = expected.get(split, 0)
        source_dir = cfg["source_base"] / split
        depth_dir = cfg["depth_base"] / split
        source_count = count_images(source_dir)
        depth_count = count_images(depth_dir)
        (gen_count, gen_dir, gen_exists), all_candidates = best_generated(cfg["generated"].get(split, []))
        gen_status = status(gen_count, target)
        if gen_status != "DONE":
            model_done = False
            unfinished.append((model, split, gen_status, gen_count, target, gen_dir))

        print(f"{split:5s} source={source_count:7d}/{target:<7d} depth={depth_count:7d}/{target:<7d} generated={gen_count:7d}/{target:<7d} {gen_status}")
        print(f"      best_generated_dir: {gen_dir}")
        for c, p, exists in all_candidates[1:]:
            if exists and c > 0:
                print(f"      other_candidate : {c:7d} {p}")
        for meta in newest_meta(Path(gen_dir), limit=2):
            t = datetime.fromtimestamp(meta.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            print(f"      meta            : {t} {meta}")

        rows.append({
            "model": model,
            "split": split,
            "expected": target,
            "source_count": source_count,
            "source_dir": str(source_dir),
            "depth_count": depth_count,
            "depth_dir": str(depth_dir),
            "generated_count": gen_count,
            "generated_status": gen_status,
            "best_generated_dir": gen_dir,
        })
    print(f"model_status: {'DONE' if model_done else 'NOT_DONE'}")

    for extra in cfg.get("extra", []):
        if extra.exists():
            print(f"extra exists: {extra} images={count_images(extra)}")
    for debug_root in cfg.get("debug_roots", []):
        if debug_root.exists():
            print(f"debug exists: {debug_root} images={count_images(debug_root)}")

print("\n============================================================")
print("Not finished")
print("============================================================")
if not unfinished:
    print("All checked generated outputs are complete.")
else:
    for model, split, st, count, target, path in unfinished:
        print(f"{model:24s} {split:5s} {st:8s} generated={count}/{target} best_dir={path}")

summary = {
    "syn_root": str(syn_root),
    "source_root": str(source_root),
    "depth_root": str(depth_root),
    "work_root": str(work_root),
    "models": models,
    "splits": splits,
    "expected": expected,
    "rows": rows,
    "unfinished": [
        {
            "model": m,
            "split": s,
            "status": st,
            "generated_count": c,
            "expected": t,
            "best_generated_dir": p,
        }
        for m, s, st, c, t, p in unfinished
    ],
}
json_path = out_dir / "synthesis_generation_completion.json"
tsv_path = out_dir / "synthesis_generation_completion.tsv"
json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
with tsv_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["model"] , delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)
print("\n============================================================")
print("Saved")
print("============================================================")
print(f"json: {json_path}")
print(f"tsv : {tsv_path}")
print("\nNotes:")
print("- watergan may be MISSING if only training/checkpoints exist and batch generation has not been run.")
print("- uwdf may be MISSING/PARTIAL while you are still tuning visual ablations.")
print("- If a model wrote outputs elsewhere, add that path to this script's generated candidates.")
PY
