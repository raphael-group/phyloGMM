#!/usr/bin/env python3
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path


def assert_nested_close(actual, expected, name, tol=0.0):
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise AssertionError(f"{name}: incompatible list values")
        for i, (a, e) in enumerate(zip(actual, expected)):
            assert_nested_close(a, e, f"{name}[{i}]", tol)
        return
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise AssertionError(f"{name}: incompatible object values")
        for key in expected:
            assert_nested_close(actual[key], expected[key], f"{name}.{key}", tol)
        return
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not math.isfinite(float(actual)) or abs(float(actual) - float(expected)) > tol:
            raise AssertionError(f"{name}: {actual} != {expected} within {tol}")
        return
    if actual != expected:
        raise AssertionError(f"{name}: {actual!r} != {expected!r}")


def run(exe, common, extra=()):
    proc = subprocess.run(
        [str(exe), *common, *extra], text=True, capture_output=True, check=True
    )
    return json.loads(proc.stdout)


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: test_run_phylo_gmm_weights.py RUN_PHYLO_GMM_EXE")
    exe = Path(sys.argv[1])

    init = {
        "Q": [[-0.6, 0.6], [0.25, -0.25]],
        "mu": [[0.0, 0.0], [3.0, 3.0]],
        "Sigma": [[[1.0, 0.1], [0.1, 1.2]], [[1.1, -0.1], [-0.1, 0.9]]],
        "pi": [0.55, 0.45],
    }

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        tree = root / "tree.nwk"
        obs = root / "obs.csv"
        params = root / "params.json"
        tree.write_text(
            "((L0:0.2,L1:0.3):0.15,(L2:0.25,(L3:0.1,L4:0.2):0.1):0.2);\n",
            encoding="utf-8",
        )
        obs.write_text(
            "cell,x0,x1\n"
            "L0,-0.2,0.1\n"
            "L1,0.3,-0.1\n"
            "L2,2.8,3.2\n"
            "L3,3.3,2.9\n"
            "L4,2.9,2.7\n",
            encoding="utf-8",
        )
        params.write_text(json.dumps(init), encoding="utf-8")

        common = [
            "--k", "2", "--tree", str(tree), "--obs", str(obs),
            "--max_iter", "4", "--tol", "0", "--n_init", "1",
            "--pi_mode", "free", "--init_params_json", str(params),
        ]
        default = run(exe, common)
        explicit_one = run(
            exe, common,
            ["--ctmc_weight", "1", "--emission_weight", "1"],
        )
        weighted = run(exe, common, ["--w_C", "0.7", "--w_E", "1.4"])
        invalid = subprocess.run(
            [str(exe), *common, "--ctmc_weight", "0"],
            text=True, capture_output=True,
        )

    # The default and explicitly supplied unit weights must take exactly the
    # same numerical path through initialization, EM, and posterior extraction.
    assert_nested_close(default, explicit_one, "unit-weight compatibility", tol=0.0)

    if weighted["objective_weights"] != {"ctmc": 0.7, "emission": 1.4}:
        raise AssertionError("weighted objective values were not echoed in JSON")
    if weighted["mode"]["ctmc_weight"] != 0.7:
        raise AssertionError("mode.ctmc_weight is incorrect")
    if weighted["mode"]["emission_weight"] != 1.4:
        raise AssertionError("mode.emission_weight is incorrect")

    for restart in weighted["per_restart"]:
        trace = restart["log_likelihood_trace"]
        for old, new in zip(trace, trace[1:]):
            if new < old - 1e-7:
                raise AssertionError(
                    f"weighted marginal objective decreased: {old} -> {new}"
                )
        for rec in restart["complete_data_objective_trace"]:
            expected = rec["root"] + 0.7 * rec["ctmc"] + 1.4 * rec["emission"]
            if abs(rec["weighted_ctmc"] - 0.7 * rec["ctmc"]) > 1e-8:
                raise AssertionError("weighted CTMC trace contribution is incorrect")
            if abs(rec["weighted_emission"] - 1.4 * rec["emission"]) > 1e-8:
                raise AssertionError("weighted emission trace contribution is incorrect")
            if abs(rec["total"] - expected) > 1e-8:
                raise AssertionError("weighted complete-data total is incorrect")

    if weighted["posteriors"] == default["posteriors"]:
        raise AssertionError("non-unit weights did not affect posterior inference")
    if invalid.returncode == 0 or "must be finite and > 0" not in invalid.stderr:
        raise AssertionError("zero CTMC weight was not rejected clearly")


if __name__ == "__main__":
    main()
