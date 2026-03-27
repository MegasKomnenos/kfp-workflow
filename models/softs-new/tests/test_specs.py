"""Tests for SOFTS experiment spec loading, Katib spec generation, and search spaces."""

from pathlib import Path

import pytest

CONFIGS_DIR = Path(__file__).parent.parent / "configs"
EXPERIMENTS_DIR = CONFIGS_DIR / "experiments"
SEARCH_SPACES_DIR = CONFIGS_DIR / "search_spaces"


# ---------------------------------------------------------------------------
# Spec loading
# ---------------------------------------------------------------------------

def test_smoke_spec_loads():
    from softs_new.specs import load_spec

    spec = load_spec(EXPERIMENTS_DIR / "fd001_smoke.yaml")
    assert spec.metadata.name == "softs-new-fd001-smoke"
    assert spec.datasets.items == ["FD001"]
    assert not spec.hpo.enabled


def test_smoke_spec_fixed_params_d_core():
    """Smoke spec must contain d_core — the SOFTS-specific STAR parameter."""
    from softs_new.specs import load_spec

    spec = load_spec(EXPERIMENTS_DIR / "fd001_smoke.yaml")
    assert "d_core" in spec.train_defaults.fixed_params
    assert spec.train_defaults.fixed_params["d_core"] == 16


def test_smoke_spec_fixed_params_no_tv_dt():
    """Smoke spec must NOT contain MambaSL-specific tv_dt parameter."""
    from softs_new.specs import load_spec

    spec = load_spec(EXPERIMENTS_DIR / "fd001_smoke.yaml")
    assert "tv_dt" not in spec.train_defaults.fixed_params


def test_default_spec_loads():
    from softs_new.specs import load_spec

    spec = load_spec(EXPERIMENTS_DIR / "fd_all_core_default.yaml")
    assert spec.metadata.name == "softs-new-fd-all-default"
    assert set(spec.datasets.items) == {"FD001", "FD002", "FD003", "FD004"}
    assert spec.hpo.enabled
    assert spec.hpo.builtin_profile == "default"


def test_aggressive_spec_loads():
    from softs_new.specs import load_spec

    spec = load_spec(EXPERIMENTS_DIR / "fd_all_core_aggressive.yaml")
    assert spec.metadata.name == "softs-new-fd-all-aggressive"
    assert spec.hpo.builtin_profile == "aggressive"
    assert spec.hpo.max_trial_count == 30


def test_all_four_stages_in_default_spec():
    from softs_new.specs import load_spec

    spec = load_spec(EXPERIMENTS_DIR / "fd_all_core_default.yaml")
    assert set(spec.stages) == {"hpo", "final_train_eval", "ablation_sweep", "aggregate_reports"}


# ---------------------------------------------------------------------------
# Katib spec generation
# ---------------------------------------------------------------------------

def test_katib_spec_contains_d_core():
    """Katib parameter manifest must include d_core for SOFTS STAR tuning."""
    from softs_new.cmapss.search_space import builtin_search_space, katib_parameter_specs

    space = builtin_search_space("default")
    katib_params = katib_parameter_specs(space)
    names = [p["name"] for p in katib_params]
    assert "d_core" in names


def test_katib_spec_excludes_mamba_params():
    """Katib spec must not contain MambaSL parameters (tv_dt, d_state, d_conv, expand)."""
    from softs_new.cmapss.search_space import builtin_search_space, katib_parameter_specs

    for profile in ("default", "aggressive"):
        space = builtin_search_space(profile)
        katib_params = katib_parameter_specs(space)
        names = [p["name"] for p in katib_params]
        for mamba_param in ("tv_dt", "tv_B", "tv_C", "use_D", "d_state", "d_conv", "expand"):
            assert mamba_param not in names, f"{mamba_param} should not appear in {profile} SOFTS search space"


def test_katib_spec_d_core_categorical():
    from softs_new.cmapss.search_space import builtin_search_space, katib_parameter_specs

    space = builtin_search_space("default")
    katib_params = katib_parameter_specs(space)
    d_core_spec = next(p for p in katib_params if p["name"] == "d_core")
    assert d_core_spec["parameterType"] == "categorical"
    assert "16" in d_core_spec["feasibleSpace"]["list"]


# ---------------------------------------------------------------------------
# Search space structure
# ---------------------------------------------------------------------------

def test_default_search_space_has_13_params():
    from softs_new.cmapss.search_space import builtin_search_space

    space = builtin_search_space("default")
    assert len(space) == 13


def test_aggressive_search_space_has_13_params():
    from softs_new.cmapss.search_space import builtin_search_space

    space = builtin_search_space("aggressive")
    assert len(space) == 13


def test_aggressive_space_has_more_or_equal_values_than_default():
    """Aggressive profile must explore at least as wide as the default."""
    from softs_new.cmapss.search_space import builtin_search_space

    default_space = {p.name: p for p in builtin_search_space("default")}
    aggressive_space = {p.name: p for p in builtin_search_space("aggressive")}

    assert set(aggressive_space.keys()) == set(default_space.keys())

    for name, agg in aggressive_space.items():
        dflt = default_space[name]
        if agg.type == "categorical" and dflt.type == "categorical":
            assert len(agg.values) >= len(dflt.values), (
                f"{name}: aggressive values ({agg.values}) must be >= default values ({dflt.values})"
            )
        elif agg.type == "log_float" and dflt.type == "log_float":
            assert agg.low <= dflt.low, f"{name}: aggressive low bound must be <= default"
            assert agg.high >= dflt.high, f"{name}: aggressive high bound must be >= default"
