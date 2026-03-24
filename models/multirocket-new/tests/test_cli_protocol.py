from argparse import _SubParsersAction

from multirocket_new.cli.main import build_parser


def _subparsers(parser):
    for action in parser._actions:
        if isinstance(action, _SubParsersAction):
            return action
    raise AssertionError("expected subparsers")


def test_top_level_command_families_match_shared_protocol():
    parser = build_parser()
    choices = set(_subparsers(parser).choices)
    assert choices == {"spec", "train", "report", "pipeline", "katib", "cluster"}


def test_shared_workflow_flags_are_present():
    parser = build_parser()
    top = _subparsers(parser).choices

    pipeline_submit = _subparsers(top["pipeline"]).choices["submit"]
    submit_flags = {option for action in pipeline_submit._actions for option in action.option_strings}
    assert {"--spec", "--namespace", "--host", "--existing-token", "--cookies"} <= submit_flags

    katib_submit = _subparsers(top["katib"]).choices["submit"]
    katib_flags = {option for action in katib_submit._actions for option in action.option_strings}
    assert {"--spec", "--dataset", "--output", "--dry-run"} <= katib_flags

    cluster_bootstrap = _subparsers(top["cluster"]).choices["bootstrap"]
    cluster_flags = {option for action in cluster_bootstrap._actions for option in action.option_strings}
    assert {"--spec", "--dry-run"} <= cluster_flags
