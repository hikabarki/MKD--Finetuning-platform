"""
ui/layout.py — Phase 3 migration anchor for the Gradio application.

The gr.Blocks() application skeleton is currently defined in app.py
(the with gr.Blocks() ... block).  Phase 3 has extracted all logic
modules; the remaining app.py reduction (wrapping into create_app())
and screen decomposition into ui/screens/*.py is deferred to Phase 4.

This module re-exports the key symbols so callers can import from here
rather than directly from app.py.
"""
from __future__ import annotations

# Phase 4 will define create_app() here with the full gr.Blocks content.
# For now, import the assembled app object from the entry point.

def create_app():
    """
    Stub — imports and returns the Gradio app assembled in app.py.

    Phase 4 will move the gr.Blocks block here so app.py becomes a
    pure entry point (~50 lines).
    """
    import app as _app_module  # noqa: PLC0415
    return _app_module.app
