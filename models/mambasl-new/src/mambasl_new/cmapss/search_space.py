from __future__ import annotations

from typing import Any, Dict, Iterable, List

from ..specs import HpoSpec, SearchParamSpec


def builtin_search_space(profile: str) -> List[SearchParamSpec]:
    if profile == "aggressive":
        return [
            SearchParamSpec(name="d_model", type="categorical", values=[32, 64, 128]),
            SearchParamSpec(name="d_state", type="categorical", values=[8, 16, 32]),
            SearchParamSpec(name="d_conv", type="categorical", values=[2, 3, 4]),
            SearchParamSpec(name="expand", type="categorical", values=[1, 2]),
            SearchParamSpec(name="num_kernels", type="categorical", values=[0, 3, 5, 7]),
            SearchParamSpec(name="tv_dt", type="categorical", values=[False, True]),
            SearchParamSpec(name="tv_B", type="categorical", values=[False, True]),
            SearchParamSpec(name="tv_C", type="categorical", values=[False, True]),
            SearchParamSpec(name="use_D", type="categorical", values=[False, True]),
            SearchParamSpec(name="projection", type="categorical", values=["last", "avg"]),
            SearchParamSpec(name="dropout", type="categorical", values=[0.0, 0.1, 0.2, 0.3, 0.4]),
            SearchParamSpec(name="batch_size", type="categorical", values=[64, 128, 256]),
            SearchParamSpec(name="lr", type="log_float", low=3e-4, high=3e-3),
            SearchParamSpec(name="weight_decay", type="log_float", low=1e-6, high=1e-3),
            SearchParamSpec(name="huber_delta", type="categorical", values=[1.0, 2.0, 5.0]),
            SearchParamSpec(name="window_size", type="categorical", values=[30, 40, 50, 60, 70]),
            SearchParamSpec(name="max_rul", type="categorical", values=[115.0, 120.0, 125.0, 130.0, 150.0]),
        ]
    return [
        SearchParamSpec(name="d_model", type="categorical", values=[32, 64, 128]),
        SearchParamSpec(name="d_state", type="categorical", values=[8, 16, 32]),
        SearchParamSpec(name="d_conv", type="categorical", values=[3, 4]),
        SearchParamSpec(name="expand", type="categorical", values=[1, 2]),
        SearchParamSpec(name="num_kernels", type="categorical", values=[0, 3, 5, 7]),
        SearchParamSpec(name="tv_dt", type="categorical", values=[False, True]),
        SearchParamSpec(name="tv_B", type="categorical", values=[False, True]),
        SearchParamSpec(name="tv_C", type="categorical", values=[False, True]),
        SearchParamSpec(name="use_D", type="categorical", values=[False, True]),
        SearchParamSpec(name="projection", type="categorical", values=["last", "avg"]),
        SearchParamSpec(name="dropout", type="categorical", values=[0.0, 0.1, 0.2, 0.3]),
        SearchParamSpec(name="batch_size", type="categorical", values=[64, 128, 256]),
        SearchParamSpec(name="lr", type="log_float", low=3e-4, high=3e-3),
        SearchParamSpec(name="weight_decay", type="log_float", low=1e-6, high=1e-3),
        SearchParamSpec(name="huber_delta", type="categorical", values=[1.0, 2.0, 5.0]),
        SearchParamSpec(name="window_size", type="categorical", values=[30, 40, 50]),
        SearchParamSpec(name="max_rul", type="categorical", values=[125.0, 130.0, 150.0]),
    ]


def resolve_search_space(hpo: HpoSpec) -> List[SearchParamSpec]:
    if hpo.search_space:
        return hpo.search_space
    return builtin_search_space(hpo.builtin_profile if hpo.builtin_profile != "custom" else "default")


def suggest_value(trial: Any, param: SearchParamSpec) -> Any:
    if param.type == "categorical":
        return trial.suggest_categorical(param.name, param.values)
    if param.type == "int":
        return trial.suggest_int(param.name, int(param.low), int(param.high), step=int(param.step or 1))
    if param.type == "float":
        return trial.suggest_float(param.name, float(param.low), float(param.high), step=param.step)
    if param.type == "log_float":
        return trial.suggest_float(param.name, float(param.low), float(param.high), log=True)
    raise ValueError(param.type)


def merge_params(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    out.update(overrides)
    return out


def katib_parameter_specs(search_space: Iterable[SearchParamSpec]) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    for param in search_space:
        if param.type == "categorical":
            specs.append(
                {
                    "name": param.name,
                    "parameterType": "categorical",
                    "feasibleSpace": {"list": [str(value) for value in param.values]},
                }
            )
        elif param.type == "int":
            space = {"min": str(int(param.low)), "max": str(int(param.high))}
            if param.step is not None:
                space["step"] = str(int(param.step))
            specs.append({"name": param.name, "parameterType": "int", "feasibleSpace": space})
        elif param.type in {"float", "log_float"}:
            space = {"min": str(float(param.low)), "max": str(float(param.high))}
            if param.step is not None and param.type == "float":
                space["step"] = str(float(param.step))
            specs.append({"name": param.name, "parameterType": "double", "feasibleSpace": space})
        else:
            raise ValueError(param.type)
    return specs
