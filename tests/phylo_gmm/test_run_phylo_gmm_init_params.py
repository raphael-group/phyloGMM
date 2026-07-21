#!/usr/bin/env python3
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path


def assert_close(actual, expected, name, tol=1e-8):
    if isinstance(expected, list):
        if not isinstance(actual, list):
            raise AssertionError(f"{name}: expected list, got {type(actual).__name__}")
        if len(actual) != len(expected):
            raise AssertionError(f"{name}: length {len(actual)} != {len(expected)}")
        for i, (a, e) in enumerate(zip(actual, expected)):
            assert_close(a, e, f"{name}[{i}]", tol)
        return

    if not math.isfinite(actual):
        raise AssertionError(f"{name}: actual value is not finite: {actual}")
    if abs(actual - expected) > tol:
        raise AssertionError(f"{name}: {actual} != {expected} within {tol}")


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: test_run_phylo_gmm_init_params.py RUN_PHYLO_GMM_EXE")

    exe = Path(sys.argv[1])
    if not exe.exists():
        raise SystemExit(f"run_phylo_gmm executable not found: {exe}")

    init = {
        "Q": [[-0.7, 0.7], [0.2, -0.2]],
        "mu": [[10.0, 11.0], [-10.0, 9.0]],
        "Sigma": [
            [[1.5, 0.2], [0.2, 1.2]],
            [[0.9, -0.1], [-0.1, 0.8]],
        ],
        "pi": [0.25, 0.75],
    }

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        tree_path = root / "tree.nwk"
        obs_path = root / "obs.csv"
        init_path = root / "init.json"

        tree_path.write_text(
            "((L0:0.3,L1:0.3):0.2,(L2:0.4,L3:0.4):0.1);\n",
            encoding="utf-8",
        )
        obs_path.write_text(
            "cell,x0,x1\n"
            "L0,0.0,0.1\n"
            "L1,0.2,-0.1\n"
            "L2,3.0,3.1\n"
            "L3,2.9,2.8\n",
            encoding="utf-8",
        )
        init_path.write_text(json.dumps(init), encoding="utf-8")

        cmd = [
            str(exe),
            "--k", "2",
            "--tree", str(tree_path),
            "--obs", str(obs_path),
            "--max_iter", "0",
            "--tol", "1e-6",
            "--n_init", "1",
            "--pi_mode", "free",
            "--init_params_json", str(init_path),
        ]
        proc = subprocess.run(cmd, text=True, capture_output=True, check=True)
        out = json.loads(proc.stdout)

    if out["mode"].get("init_params_json") != str(init_path):
        raise AssertionError("mode.init_params_json did not echo the init path")
    if len(out["per_restart"]) != 2:
        raise AssertionError(f"expected 2 restarts, got {len(out['per_restart'])}")

    for idx, restart in enumerate(out["per_restart"]):
        initial = restart["initial"]
        assert_close(initial["Q"], init["Q"], f"restart[{idx}].initial.Q")
        assert_close(initial["mu"], init["mu"], f"restart[{idx}].initial.mu")
        assert_close(initial["Sigma"], init["Sigma"], f"restart[{idx}].initial.Sigma")
        assert_close(initial["pi"], init["pi"], f"restart[{idx}].initial.pi")
        if restart["iterations"] != 0:
            raise AssertionError(
                f"restart[{idx}] should not run EM iterations under --max_iter 0"
            )


if __name__ == "__main__":
    main()
