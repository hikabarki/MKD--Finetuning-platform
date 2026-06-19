"""Extended tests for ui/components.py — data listers and HTML panels."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch


@patch("ui.components.ROOT", None)   # replaced per-test via monkeypatch
def _placeholder(): pass


def test_list_datasets_empty(tmp_project, monkeypatch):
    import ui.components as comp
    monkeypatch.setattr(comp, "ROOT", tmp_project)
    assert comp.list_datasets() == []


def test_list_datasets_with_dirs(tmp_project, monkeypatch):
    import ui.components as comp
    monkeypatch.setattr(comp, "ROOT", tmp_project)
    (tmp_project / "datasets" / "ds1").mkdir()
    (tmp_project / "datasets" / "ds2").mkdir()
    result = comp.list_datasets()
    assert set(result) == {"ds1", "ds2"}


def test_list_cleaned_datasets(tmp_project, monkeypatch):
    import ui.components as comp
    monkeypatch.setattr(comp, "ROOT", tmp_project)
    d = tmp_project / "datasets" / "cleaned_ds"
    d.mkdir()
    (d / "train.jsonl").write_text("")
    result = comp.list_cleaned_datasets()
    assert "cleaned_ds" in result


def test_list_models_empty(tmp_project, monkeypatch):
    import ui.components as comp
    monkeypatch.setattr(comp, "ROOT", tmp_project)
    assert comp.list_models() == []


def test_list_models_with_dir(tmp_project, monkeypatch):
    import ui.components as comp
    monkeypatch.setattr(comp, "ROOT", tmp_project)
    (tmp_project / "models" / "mymodel").mkdir()
    assert "mymodel" in comp.list_models()


def test_list_checkpoints(tmp_project, monkeypatch):
    import ui.components as comp
    monkeypatch.setattr(comp, "ROOT", tmp_project)
    (tmp_project / "checkpoints" / "run_001").mkdir()
    result = comp.list_checkpoints()
    assert "run_001" in result


def test_list_exports(tmp_project, monkeypatch):
    import ui.components as comp
    monkeypatch.setattr(comp, "ROOT", tmp_project)
    (tmp_project / "exports" / "run_001").mkdir()
    result = comp.list_exports()
    assert "run_001" in result


def test_get_pipeline_html_returns_str(tmp_project, monkeypatch):
    import ui.components as comp
    monkeypatch.setattr(comp, "ROOT", tmp_project)
    html = comp.get_pipeline_html()
    assert isinstance(html, str)
    assert "PIPELINE" in html


def test_get_stats_html_returns_str(tmp_project, monkeypatch):
    import ui.components as comp
    monkeypatch.setattr(comp, "ROOT", tmp_project)
    html = comp.get_stats_html()
    assert isinstance(html, str)
    assert "Datasets" in html


def test_get_guide_html_returns_str(tmp_project, monkeypatch):
    import ui.components as comp
    monkeypatch.setattr(comp, "ROOT", tmp_project)
    html = comp.get_guide_html()
    assert isinstance(html, str)
    assert "Dataset" in html


def test_get_gpu_html_unavailable(tmp_project, monkeypatch):
    """When nvidia-smi is absent, get_gpu_html returns an error message."""
    import ui.components as comp
    with patch("subprocess.check_output", side_effect=FileNotFoundError("nvidia-smi")):
        html = comp.get_gpu_html()
    assert "unavailable" in html.lower() or "GPU" in html


def test_get_state_html(tmp_project, monkeypatch):
    import ui.components as comp
    monkeypatch.setattr(comp, "ROOT", tmp_project)
    html = comp.get_state_html()
    assert isinstance(html, str)


def test_next_step_card_unknown_command(tmp_project, monkeypatch):
    import ui.components as comp
    monkeypatch.setattr(comp, "ROOT", tmp_project)
    html = comp.next_step_card("unknown-command")
    assert html == ""


def test_next_step_card_known_command(tmp_project, monkeypatch):
    import ui.components as comp
    monkeypatch.setattr(comp, "ROOT", tmp_project)
    # Use any command key from _NEXT_TAB if it exists
    if comp._NEXT_TAB:
        key = next(iter(comp._NEXT_TAB))
        html = comp.next_step_card(key)
        assert isinstance(html, str)


def test_read_sample_no_file(tmp_project, monkeypatch):
    import ui.components as comp
    monkeypatch.setattr(comp, "ROOT", tmp_project)
    result = comp.read_sample("nonexistent")
    assert isinstance(result, str)


def test_read_log_no_file(tmp_project, monkeypatch):
    import ui.components as comp
    monkeypatch.setattr(comp, "ROOT", tmp_project)
    result = comp.read_log("nonexistent")
    assert isinstance(result, str)


def test_read_eval_summary_no_file(tmp_project, monkeypatch):
    import ui.components as comp
    monkeypatch.setattr(comp, "ROOT", tmp_project)
    result = comp.read_eval_summary("nonexistent")
    assert isinstance(result, str)


def test_read_data_card_no_file(tmp_project, monkeypatch):
    import ui.components as comp
    monkeypatch.setattr(comp, "ROOT", tmp_project)
    result = comp.read_data_card("nonexistent")
    assert isinstance(result, str)


def test_read_model_card_no_file(tmp_project, monkeypatch):
    import ui.components as comp
    monkeypatch.setattr(comp, "ROOT", tmp_project)
    result = comp.read_model_card("nonexistent")
    assert isinstance(result, str)


def test_read_config_file_no_file(tmp_project, monkeypatch):
    import ui.components as comp
    monkeypatch.setattr(comp, "ROOT", tmp_project)
    result = comp.read_config_file("nonexistent.yaml")
    assert isinstance(result, str)
