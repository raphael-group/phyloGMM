#!/usr/bin/env python3
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path


PARAMETER_FIELDS = ("Q", "mu", "Sigma", "pi")


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


def command(exe, tree_path, obs_path, k, max_iter=0):
    return [
        str(exe),
        "--k", str(k),
        "--tree", str(tree_path),
        "--obs", str(obs_path),
        "--max_iter", str(max_iter),
        "--tol", "1e-6",
        "--q_init_rate", "0.37",
        "--n_init", "1",
        "--pi_mode", "free",
    ]


def run_fit(
    exe, tree_path, obs_path, k, init_path=None, max_iter=0, extra_args=()
):
    cmd = command(exe, tree_path, obs_path, k, max_iter=max_iter)
    if init_path is not None:
        cmd.extend(["--init_params_json", str(init_path)])
    cmd.extend(extra_args)
    proc = subprocess.run(cmd, text=True, capture_output=True, check=True)
    return json.loads(proc.stdout)


def restarts_by_identity(output):
    return {(r["label"], r["seed"]): r for r in output["per_restart"]}


def assert_partial_overlay(output, baseline, expected, init_path, case_name):
    if output["mode"].get("init_params_json") != str(init_path):
        raise AssertionError(f"{case_name}: mode.init_params_json did not echo path")

    actual_restarts = restarts_by_identity(output)
    baseline_restarts = restarts_by_identity(baseline)
    if actual_restarts.keys() != baseline_restarts.keys():
        raise AssertionError(f"{case_name}: restart identities changed")

    for identity, restart in actual_restarts.items():
        initial = restart["initial"]
        baseline_initial = baseline_restarts[identity]["initial"]
        for field in PARAMETER_FIELDS:
            wanted = expected[field] if field in expected else baseline_initial[field]
            assert_close(
                initial[field], wanted, f"{case_name}.{identity}.initial.{field}"
            )
        if restart["iterations"] != 0:
            raise AssertionError(f"{case_name}.{identity}: expected zero EM iterations")


def assert_invalid_init(exe, tree_path, obs_path, init_path, message):
    proc = subprocess.run(
        command(exe, tree_path, obs_path, 2)
        + ["--init_params_json", str(init_path)],
        text=True,
        capture_output=True,
    )
    if proc.returncode == 0:
        raise AssertionError(f"{init_path.name}: invalid initialization was accepted")
    if message not in proc.stderr:
        raise AssertionError(
            f"{init_path.name}: expected stderr to contain {message!r}; "
            f"got {proc.stderr!r}"
        )


def assert_covariance_only_freeze(output, expected_sigma, case_name):
    if output["mode"].get("fix_covariance") is not True:
        raise AssertionError(f"{case_name}: fix_covariance mode was not reported")
    if output["mode"].get("fix_emission") is True:
        raise AssertionError(f"{case_name}: covariance-only mode reported fix_emission")

    assert_close(output["Sigma"], expected_sigma, f"{case_name}.Sigma")
    for identity, restart in restarts_by_identity(output).items():
        initial = restart["initial"]
        final = restart["final"]
        assert_close(
            initial["Sigma"], expected_sigma,
            f"{case_name}.{identity}.initial.Sigma",
        )
        assert_close(
            final["Sigma"], initial["Sigma"],
            f"{case_name}.{identity}.final.Sigma",
            tol=0.0,
        )

        mean_shift = sum(
            abs(new - old)
            for old_row, new_row in zip(initial["mu"], final["mu"])
            for old, new in zip(old_row, new_row)
        )
        if mean_shift <= 1e-6:
            raise AssertionError(
                f"{case_name}.{identity}: means did not update under "
                "--fix_covariance"
            )
        if restart["iterations"] != 1:
            raise AssertionError(
                f"{case_name}.{identity}: expected one EM iteration"
            )
        trace = restart["log_likelihood_trace"]
        if len(trace) != 2 or not all(math.isfinite(value) for value in trace):
            raise AssertionError(
                f"{case_name}.{identity}: invalid likelihood trace {trace}"
            )
        if trace[1] < trace[0] - 1e-8:
            raise AssertionError(
                f"{case_name}.{identity}: likelihood decreased {trace}"
            )


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: test_run_phylo_gmm_init_params.py RUN_PHYLO_GMM_EXE")

    exe = Path(sys.argv[1])
    if not exe.exists():
        raise SystemExit(f"run_phylo_gmm executable not found: {exe}")

    full_init = {
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

        baseline = run_fit(exe, tree_path, obs_path, 2)
        if len(baseline["per_restart"]) != 2:
            raise AssertionError(
                f"baseline: expected 2 restarts, got {len(baseline['per_restart'])}"
            )

        cases = [
            ("full", full_init, full_init),
            ("q_only", {"Q": full_init["Q"]}, {"Q": full_init["Q"]}),
            ("pi_only", {"pi": [1.0, 3.0]}, {"pi": [0.25, 0.75]}),
            ("mu_only", {"mu": full_init["mu"]}, {"mu": full_init["mu"]}),
            (
                "sigma_only",
                {"Sigma": full_init["Sigma"]},
                {"Sigma": full_init["Sigma"]},
            ),
            (
                "sigma_diag_only",
                {"sigma_diag": [[1.5, 1.2], [0.9, 0.8]]},
                {
                    "Sigma": [
                        [[1.5, 0.0], [0.0, 1.2]],
                        [[0.9, 0.0], [0.0, 0.8]],
                    ]
                },
            ),
        ]

        for case_name, payload, expected in cases:
            init_path = root / f"{case_name}.json"
            init_path.write_text(json.dumps(payload), encoding="utf-8")
            output = run_fit(exe, tree_path, obs_path, 2, init_path)
            assert_partial_overlay(output, baseline, expected, init_path, case_name)

        covariance_init = {
            "Q": full_init["Q"],
            "mu": [[-0.5, 0.4], [3.5, 2.5]],
            "Sigma": full_init["Sigma"],
            "pi": full_init["pi"],
        }
        covariance_path = root / "fix_covariance.json"
        covariance_path.write_text(json.dumps(covariance_init), encoding="utf-8")

        full_covariance_output = run_fit(
            exe, tree_path, obs_path, 2,
            init_path=covariance_path,
            max_iter=1,
            extra_args=["--fix_covariance"],
        )
        assert_covariance_only_freeze(
            full_covariance_output,
            covariance_init["Sigma"],
            "full_covariance",
        )

        diagonal_sigma = [
            [[matrix[0][0], 0.0], [0.0, matrix[1][1]]]
            for matrix in covariance_init["Sigma"]
        ]
        diagonal_covariance_output = run_fit(
            exe, tree_path, obs_path, 2,
            init_path=covariance_path,
            max_iter=1,
            extra_args=["--fix_covariance", "--diag_covariance"],
        )
        assert_covariance_only_freeze(
            diagonal_covariance_output,
            diagonal_sigma,
            "diagonal_covariance",
        )

        # A complete emission specification must continue to bypass the N < K
        # guard in the data-driven initializers. This was supported before
        # arbitrary partial overlays and is useful for fully external starts.
        small_tree = root / "small_tree.nwk"
        small_obs = root / "small_obs.csv"
        n_lt_k_init = root / "n_lt_k_complete.json"
        small_tree.write_text("(A:0.1,B:0.1);\n", encoding="utf-8")
        small_obs.write_text("cell,x0\nA,0.0\nB,1.0\n", encoding="utf-8")
        n_lt_k_payload = {
            "mu": [[0.0], [0.5], [1.0]],
            "Sigma": [[[0.2]], [[0.2]], [[0.2]]],
        }
        n_lt_k_init.write_text(json.dumps(n_lt_k_payload), encoding="utf-8")
        n_lt_k_output = run_fit(exe, small_tree, small_obs, 3, n_lt_k_init)
        for restart in n_lt_k_output["per_restart"]:
            assert_close(
                restart["initial"]["mu"], n_lt_k_payload["mu"], "n_lt_k.initial.mu"
            )
            assert_close(
                restart["initial"]["Sigma"],
                n_lt_k_payload["Sigma"],
                "n_lt_k.initial.Sigma",
            )

        invalid_cases = [
            ("empty.json", {}, "expected at least one of"),
            ("unknown.json", {"muu": full_init["mu"]}, "expected at least one of"),
            ("empty_q.json", {"Q": []}, "Q: expected 2 rows, got 0"),
        ]
        for filename, payload, message in invalid_cases:
            init_path = root / filename
            init_path.write_text(json.dumps(payload), encoding="utf-8")
            assert_invalid_init(exe, tree_path, obs_path, init_path, message)


if __name__ == "__main__":
    main()
