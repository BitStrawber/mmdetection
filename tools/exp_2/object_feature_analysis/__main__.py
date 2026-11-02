from __future__ import annotations

import sys
from importlib import import_module


COMMANDS = {
    "sample-instances": (".sampling", "main"),
    "extract-roi": (".extract_roi_features", "main"),
    "compute-metrics": (".metrics", "main"),
    "render-figures": (".render_figures", "main"),
    "run-all": (".run_pipeline", "main"),
    "self-check": (".validate_mapping", "main"),
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print("Usage: python -m tools.exp_2.object_feature_analysis COMMAND [ARGS]")
        print("Commands:")
        for command in COMMANDS:
            print(f"  {command}")
        return
    command = sys.argv.pop(1)
    if command not in COMMANDS:
        raise SystemExit(f"Unknown command: {command}")
    module_name, function_name = COMMANDS[command]
    module = import_module(module_name, package=__package__)
    getattr(module, function_name)()


if __name__ == "__main__":
    main()
