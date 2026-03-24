"""Project-level exceptions for the HPO engine.

Plugins raise these instead of Optuna types, keeping Optuna out of plugin
code and preserving the project → plugin dependency direction.
"""


class TrialPruned(Exception):
    """Raised by a plugin's ``hpo_objective`` to signal that a trial should
    be pruned (e.g. insufficient data for the candidate window size)."""
