"""Regression tests for KFP v2 component serialization safety.

KFP v2 serializes ONLY the body of @dsl.component-decorated functions into
an isolated ephemeral_component.py script that runs inside the pod. Module-
level definitions — helper functions, module imports, constants — are NOT
captured and will be absent at pod runtime.

These tests statically check every @dsl.component function in this project
for free-name violations: names referenced inside the function body that are
not locally defined via import, assignment, parameter, or nested def.

Root incident: run 7b9d21fb (mrhysp-cmapss-smoke) failed with
  NameError: name '_resolve_data_mount_path_from_spec' is not defined
because the helper was defined at module level in components/load_data.py
but called inside load_data_component.
"""
from __future__ import annotations

import ast
import builtins
import inspect
import textwrap

import pytest

from kfp_workflow.components.evaluate import evaluate_component
from kfp_workflow.components.load_data import load_data_component
from kfp_workflow.components.preprocess import preprocess_component
from kfp_workflow.components.save_model import save_model_component
from kfp_workflow.components.train import train_component
from kfp_workflow.benchmark.components import (
    cleanup_benchmark_model_component,
    deploy_benchmark_model_component,
    run_benchmark_component,
    wait_for_benchmark_model_component,
)

ALL_COMPONENTS = [
    load_data_component,
    preprocess_component,
    train_component,
    evaluate_component,
    save_model_component,
    deploy_benchmark_model_component,
    wait_for_benchmark_model_component,
    run_benchmark_component,
    cleanup_benchmark_model_component,
]

_BUILTINS: frozenset[str] = frozenset(dir(builtins))


class _FreeNameChecker(ast.NodeVisitor):
    """AST visitor that collects names used but not locally bound in a function.

    Handles:
    - import / import-from (bind the top-level name or asname)
    - assignment, augmented assignment, annotated assignment
    - for-loop targets, comprehension iteration targets (walrus operator)
    - nested function / class definitions (bind the def/class name; push scope)
    - function parameters (args, *args, **kwargs, keyword-only args)
    - comprehension scopes (each generator expression pushes its own scope)

    Does NOT attempt full data-flow analysis; the goal is to catch the
    specific pattern of calling a module-level helper from inside a component.
    """

    def __init__(self) -> None:
        # Stack of bound-name sets; index 0 is the outermost function scope.
        self._scopes: list[set[str]] = [set()]
        self.free: set[str] = set()

    # ------------------------------------------------------------------ scopes

    def _all_bound(self) -> set[str]:
        result: set[str] = set()
        for s in self._scopes:
            result |= s
        return result

    def _bind(self, name: str) -> None:
        self._scopes[-1].add(name)

    def _push(self) -> None:
        self._scopes.append(set())

    def _pop(self) -> None:
        self._scopes.pop()

    # ---------------------------------------------------------------- visitors

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # Bind the function name in the enclosing scope
        self._bind(node.name)
        self._push()
        # Bind all parameter names
        all_args = (
            node.args.args
            + node.args.posonlyargs
            + node.args.kwonlyargs
        )
        for arg in all_args:
            self._bind(arg.arg)
        if node.args.vararg:
            self._bind(node.args.vararg.arg)
        if node.args.kwarg:
            self._bind(node.args.kwarg.arg)
        self.generic_visit(node)
        self._pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._bind(node.name)
        self._push()
        self.generic_visit(node)
        self._pop()

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name.split(".")[0]
            self._bind(name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            self._bind(name)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._collect_stores(target)
        self.visit(node.value)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._collect_stores(node.target)
        self.visit(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._collect_stores(node.target)
        if node.value:
            self.visit(node.value)

    def visit_For(self, node: ast.For) -> None:
        self._collect_stores(node.target)
        self.generic_visit(node)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self._bind(node.target.id)
        self.visit(node.value)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name:
            self._bind(node.name)
        self.generic_visit(node)

    # Comprehensions get their own scope for their iteration variables
    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._push()
        self.generic_visit(node)
        self._pop()

    visit_SetComp = visit_ListComp
    visit_DictComp = visit_ListComp
    visit_GeneratorExp = visit_ListComp

    def visit_comprehension(self, node: ast.comprehension) -> None:
        self._collect_stores(node.target)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            if node.id not in self._all_bound() and node.id not in _BUILTINS:
                self.free.add(node.id)

    # ---------------------------------------------------------------- helpers

    def _collect_stores(self, node: ast.expr) -> None:
        if isinstance(node, ast.Name):
            self._bind(node.id)
        elif isinstance(node, (ast.Tuple, ast.List)):
            for elt in node.elts:
                self._collect_stores(elt)
        elif isinstance(node, ast.Starred):
            self._collect_stores(node.value)


def _free_names_in_component(component) -> set[str]:
    """Return names referenced but not locally bound in a @dsl.component func."""
    fn = component.python_func
    src = inspect.getsource(fn)
    lines = src.splitlines()
    # Strip any decorator lines so ast.parse sees a bare 'def ...'
    def_index = next(
        i for i, line in enumerate(lines) if line.lstrip().startswith("def ")
    )
    body_src = textwrap.dedent("\n".join(lines[def_index:]))
    tree = ast.parse(body_src)
    fn_node = tree.body[0]
    assert isinstance(fn_node, ast.FunctionDef)

    checker = _FreeNameChecker()
    # Visit only the body statements; the top-level function parameters are
    # already handled by visit_FunctionDef but we need to bootstrap from here.
    checker.visit(fn_node)
    return checker.free


@pytest.mark.parametrize("component", ALL_COMPONENTS, ids=lambda c: c.name)
def test_component_has_no_free_names(component):
    """No @dsl.component function may reference names outside its own body.

    KFP v2 serializes only the function body; module-level helpers are absent
    at pod runtime. Any name used inside the function must be locally bound
    (via import, assignment, parameter, or nested def).

    Regression guard for incident run 7b9d21fb.
    """
    free = _free_names_in_component(component)
    assert not free, (
        f"Component '{component.name}' references names that are not locally "
        f"defined and will cause NameError at pod runtime: {sorted(free)}\n\n"
        "All imports, constants, and helper functions used at runtime MUST be "
        "defined inside the @dsl.component function body."
    )
