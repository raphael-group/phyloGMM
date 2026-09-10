#!/usr/bin/env python3
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path


def assert_nested_close(actual, expected, name, tol=2e-8):
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
        if not math.isfinite(float(actual)):
            raise AssertionError(f"{name}: non-finite value {actual}")
        if abs(float(actual) - float(expected)) > tol:
            raise AssertionError(
                f"{name}: {actual} != {expected} within tolerance {tol}"
            )
        return
    if actual != expected:
        raise AssertionError(f"{name}: {actual!r} != {expected!r}")


def run(exe, args):
    proc = subprocess.run(
        [str(exe), *args], text=True, capture_output=True, check=True
    )
    return json.loads(proc.stdout)


def test_root_suffix_is_ignored(exe, root):
    params = {
        "Q": [[-0.7, 0.7], [0.2, -0.2]],
        "mu": [[0.0], [2.0]],
        "Sigma": [[[0.8]], [[1.1]]],
        # Deliberately not the stationary distribution of Q.
        "pi": [0.8, 0.2],
    }
    params_path = root / "root_suffix_params.json"
    obs_path = root / "root_suffix_obs.csv"
    params_path.write_text(json.dumps(params), encoding="utf-8")
    obs_path.write_text(
        "cell,x0\n"
        "L0,0.25\n"
        "L1,1.20\n",
        encoding="utf-8",
    )

    # The epsilon case represented the intended no-stem likelihood before the
    # fix, while omitted/:0/:1 were all converted to a unit stem by the parser.
    suffixes = {
        "omitted": "",
        "zero": ":0",
        "one": ":1",
        "large": ":5",
        "epsilon": ":1e-12",
    }
    outputs = {}
    for label, suffix in suffixes.items():
        tree_path = root / f"root_suffix_{label}.nwk"
        tree_path.write_text(
            f"(L0:0.2,L1:0.35)R{suffix};\n", encoding="utf-8"
        )
        outputs[label] = run(
            exe,
            [
                "--k", "2",
                "--tree", str(tree_path),
                "--obs", str(obs_path),
                "--max_iter", "0",
                "--tol", "0",
                "--n_init", "1",
                "--pi_mode", "free",
                "--init_params_json", str(params_path),
            ],
        )

    fields = (
        "log_likelihood",
        "Q",
        "mu",
        "Sigma",
        "pi",
        "posteriors",
        "internal_node_posteriors",
        "expected_transitions",
        "expected_dwell_times",
        "per_restart",
    )
    reference = outputs["epsilon"]
    for label in ("omitted", "zero", "one", "large"):
        for field in fields:
            assert_nested_close(
                outputs[label][field],
                reference[field],
                f"root suffix {label}.{field}",
            )


def test_free_pi_em_is_coherent(exe, root):
    params = {
        "Q": [[-0.5, 0.5], [0.5, -0.5]],
        "mu": [[-5.0], [5.0]],
        "Sigma": [[[0.01]], [[0.01]]],
        "pi": [0.99, 0.01],
    }
    params_path = root / "free_pi_params.json"
    params_path.write_text(json.dumps(params), encoding="utf-8")

    args = ["--k", "2"]
    for tree_index in range(5):
        leaf_a = f"A{tree_index}"
        leaf_b = f"B{tree_index}"
        tree_path = root / f"free_pi_tree_{tree_index}.nwk"
        obs_path = root / f"free_pi_obs_{tree_index}.csv"
        tree_path.write_text(
            f"({leaf_a}:1e-6,{leaf_b}:1e-6)R{tree_index};\n",
            encoding="utf-8",
        )
        observed_value = -5.0 if tree_index < 4 else 5.0
        obs_path.write_text(
            "cell,x0\n"
            f"{leaf_a},{observed_value}\n"
            f"{leaf_b},{observed_value}\n",
            encoding="utf-8",
        )
        args.extend(["--tree", str(tree_path), "--obs", str(obs_path)])

    args.extend(
        [
            "--max_iter", "1",
            "--tol", "0",
            "--n_init", "1",
            "--pi_mode", "free",
            "--init_params_json", str(params_path),
            "--fix_q_json", str(params_path),
            "--fix_emission",
        ]
    )
    output = run(exe, args)

    for restart_index, restart in enumerate(output["per_restart"]):
        trace = restart["log_likelihood_trace"]
        if len(trace) != 2:
            raise AssertionError(
                f"restart {restart_index}: expected one EM update, got {trace}"
            )
        if trace[1] < trace[0] - 1e-8:
            raise AssertionError(
                f"restart {restart_index}: free-pi EM decreased likelihood: {trace}"
            )
        fitted_pi0 = restart["final"]["pi"][0]
        if abs(fitted_pi0 - 0.8) > 1e-5:
            raise AssertionError(
                f"restart {restart_index}: unexpected fitted pi "
                f"{restart['final']['pi']}"
            )


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: test_run_phylo_gmm_root_stem.py RUN_PHYLO_GMM_EXE")
    exe = Path(sys.argv[1])
    if not exe.exists():
        raise SystemExit(f"run_phylo_gmm executable not found: {exe}")

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        test_root_suffix_is_ignored(exe, root)
        test_free_pi_em_is_coherent(exe, root)


if __name__ == "__main__":
    main()
