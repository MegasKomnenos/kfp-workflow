from __future__ import annotations

from typing import Any, Dict, Iterable, List

from .specs import HpoSpec, SearchParamSpec


def builtin_search_space(profile: str) -> List[SearchParamSpec]:
    if profile == "aggressive":
        return [
            SearchParamSpec(name="mr_num_kernels", type="int", low=84, high=840, step=84),
            SearchParamSpec(name="n_kernels", type="int", low=1, high=6, step=1),
            SearchParamSpec(name="n_groups", type="int", low=16, high=160, step=16),
            SearchParamSpec(name="n_kernels_sp", type="int", low=64, high=640, step=64),
            SearchParamSpec(name="seed", type="categorical", values=[7, 13, 21, 42, 84]),
        ]
    return [
        SearchParamSpec(name="mr_num_kernels", type="int", low=84, high=588, step=84),
        SearchParamSpec(name="n_kernels", type="int", low=1, high=4, step=1),
        SearchParamSpec(name="n_groups", type="int", low=16, high=128, step=16),
        SearchParamSpec(name="n_kernels_sp", type="int", low=64, high=512, step=64),
        SearchParamSpec(name="seed", type="categorical", values=[7, 13, 21, 42]),
    ]


def resolve_search_space(hpo: HpoSpec) -> List[SearchParamSpec]:
    if hpo.search_space:
        return hpo.search_space
    return builtin_search_space(hpo.builtin_profile if hpo.builtin_profile != "custom" else "default")


def katib_parameter_specs(search_space: Iterable[SearchParamSpec]) -> List[Dict[str, Any]]:
    out = []
    for param in search_space:
        if param.type == "categorical":
            feasible = {"list": [str(value) for value in param.values or []]}
            parameter_type = "categorical"
        else:
            feasible = {"min": str(param.low), "max": str(param.high)}
            if param.step is not None:
                feasible["step"] = str(param.step)
            parameter_type = "double" if param.type in {"float", "log_float"} else "int"
        item = {
            "name": param.name,
            "parameterType": parameter_type,
            "feasibleSpace": feasible,
        }
        if param.type == "log_float":
            item["scaleType"] = "log"
        out.append(item)
    return out
