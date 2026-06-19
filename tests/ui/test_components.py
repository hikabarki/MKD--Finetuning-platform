"""Tests for ui/components.py — ANSI renderer and HTML helpers."""
from __future__ import annotations


def test_ansi_html_escape():
    """HTML special chars must be escaped before ANSI processing."""
    from ui.components import ansi_to_html
    result = ansi_to_html("<script>alert(1)</script>")
    assert "<script>" not in result
    assert "&lt;script&gt;" in result


def test_ansi_to_html_plain_text():
    from ui.components import ansi_to_html
    assert ansi_to_html("hello world") == "hello world"


def test_ansi_to_html_green():
    from ui.components import ansi_to_html
    result = ansi_to_html("\033[32mgreen\033[0m")
    assert "color:#3fb950" in result
    assert "green" in result
    assert "</span>" in result


def test_ansi_to_html_red():
    from ui.components import ansi_to_html
    result = ansi_to_html("\033[31merror\033[0m")
    assert "color:#f85149" in result


def test_ansi_to_html_bold():
    from ui.components import ansi_to_html
    result = ansi_to_html("\033[1mbold\033[0m")
    assert "font-weight:bold" in result


def test_ansi_to_html_unclosed_span():
    """If ANSI sequence is never reset, span is auto-closed."""
    from ui.components import ansi_to_html
    result = ansi_to_html("\033[32mno reset")
    assert result.count("<span") == result.count("</span>")


def test_ansi_to_html_newline():
    from ui.components import ansi_to_html
    result = ansi_to_html("line1\nline2")
    assert "<br>" in result


def test_ansi_to_html_ampersand():
    from ui.components import ansi_to_html
    result = ansi_to_html("a & b")
    assert "&amp;" in result
    assert " & " not in result


def test_terminal_wrap_contains_content():
    from ui.components import terminal_wrap
    html = terminal_wrap("some output")
    assert "some output" in html


def test_terminal_wrap_height():
    from ui.components import terminal_wrap
    html = terminal_wrap("x", height=500)
    assert "500" in html


def test_terminal_wrap_default_height():
    from ui.components import terminal_wrap
    html = terminal_wrap("x")
    assert "380" in html
