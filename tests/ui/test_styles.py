"""Tests for ui/styles.py — CSS constant."""
from __future__ import annotations


def test_css_constant_is_string():
    import ui.styles as styles
    assert isinstance(styles.CSS, str)


def test_css_contains_footer_hide():
    import ui.styles as styles
    assert "footer" in styles.CSS


def test_css_contains_button_rules():
    import ui.styles as styles
    assert "button" in styles.CSS
