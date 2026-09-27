"""
Real pytest tests running the fast/small pipeline end-to-end and asserting
genuine properties of the results (not mocked).
"""

import os
import sys
import json
import subprocess

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_pipeline_once():
    """Run the full pipeline once (fast: small models, 500-sample test
    set) so all downstream tests can inspect real generated artifacts."""
    subprocess.run([sys.executable, "generate_molecular_conditions.py"], cwd=BASE, check=True)
    subprocess.run([sys.executable, "train_architectures.py"], cwd=BASE, check=True)
    subprocess.run([sys.executable, "run_robustness_evaluation.py"], cwd=BASE, check=True)
    subprocess.run([sys.executable, "test_proposed_fix.py"], cwd=BASE, check=True)


def setup_module(module):
    run_pipeline_once()


def test_12_architectures_trained():
    with open(os.path.join(BASE, "models", "architectures_meta.json")) as f:
        meta = json.load(f)
    assert len(meta) == 12
    for m in meta:
        assert os.path.exists(os.path.join(BASE, "models", f"{m['id']}.pkl"))


def test_95_conditions_present():
    with open(os.path.join(BASE, "data", "conditions_meta.json")) as f:
        meta = json.load(f)
    assert len(meta) == 95
    for c in meta:
        assert os.path.exists(os.path.join(BASE, "data", "conditions", f"{c['name']}.npz"))


def test_full_matrix_no_missing_entries():
    import pandas as pd
    df = pd.read_csv(os.path.join(BASE, "output", "eval_matrix.csv"))
    assert len(df) == 12 * 95
    assert df["accuracy"].isna().sum() == 0
    assert df["architecture"].nunique() == 12
    assert df["condition"].nunique() == 95


def test_at_least_one_failure_mode_identified():
    with open(os.path.join(BASE, "output", "findings.json")) as f:
        payload = json.load(f)
    findings = payload["findings"]
    # a genuine failure mode = some architecture scoring meaningfully below
    # the pass threshold on at least one condition category
    assert findings["worst_architecture"]["accuracy"] < 0.95
    assert findings["failure_rate_by_category"][findings["worst_category"]] > 0


def test_proposed_fix_runs_and_reports_real_before_after():
    with open(os.path.join(BASE, "output", "fix_test_result.json")) as f:
        result = json.load(f)
    assert "before_mean_accuracy" in result
    assert "after_mean_accuracy" in result
    assert 0.0 <= result["before_mean_accuracy"] <= 1.0
    assert 0.0 <= result["after_mean_accuracy"] <= 1.0
    assert os.path.exists(os.path.join(BASE, "models", "arch_13_ensemble_fix.pkl"))
    # not asserting improvement is guaranteed -- just that a real number was produced
    assert isinstance(result["genuinely_improved"], bool)
