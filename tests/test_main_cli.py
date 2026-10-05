import pytest
from main import build_parser


def test_cli_mode_arguments():
    parser = build_parser()
    args_default = parser.parse_args([])
    assert args_default.mode == "deterministic"

    args_freeform = parser.parse_args(["--mode", "freeform"])
    assert args_freeform.mode == "freeform"
