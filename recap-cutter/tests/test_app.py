import pytest

gr = pytest.importorskip("gradio")


def test_app_builds():
    from recapcut import app

    demo = app.build_ui()
    assert isinstance(demo, gr.Blocks)
    assert "ffmpeg" in app.status_markdown()


def test_app_command_registered():
    from recapcut import cli

    args = cli.build_parser().parse_args(["app", "--port", "8000", "--no-browser"])
    assert args.port == 8000 and args.no_browser and args.func is cli.cmd_app
