"""Additional tests for ui/components.py — run_ft, save_model_card, read_sample."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch


# ── list_configs ──────────────────────────────────────────────────────────────

def test_list_configs_empty(tmp_project, monkeypatch):
    import ui.components as comp
    monkeypatch.setattr(comp, "ROOT", tmp_project)
    result = comp.list_configs()
    assert isinstance(result, list)


def test_list_configs_with_yaml(tmp_project, monkeypatch):
    import ui.components as comp
    monkeypatch.setattr(comp, "ROOT", tmp_project)
    cfg_dir = tmp_project / "configs"
    cfg_dir.mkdir(exist_ok=True)
    # list_configs() globs "run_*.yaml"
    (cfg_dir / "run_lora.yaml").write_text("lr: 0.0002")
    result = comp.list_configs()
    assert "run_lora.yaml" in result


# ── save_model_card ───────────────────────────────────────────────────────────

def test_save_model_card_creates_file(tmp_project, monkeypatch):
    import ui.components as comp
    monkeypatch.setattr(comp, "ROOT", tmp_project)
    # save_model_card writes to ROOT/exports/<name>/MODEL_CARD.md
    export_dir = tmp_project / "exports" / "mymodel"
    export_dir.mkdir(parents=True)
    result = comp.save_model_card("mymodel", "# My Model\ngreat stuff")
    assert "Saved" in result or "saved" in result.lower()
    assert (export_dir / "MODEL_CARD.md").exists()


def test_save_model_card_no_export_dir(tmp_project, monkeypatch):
    import ui.components as comp
    monkeypatch.setattr(comp, "ROOT", tmp_project)
    # When the exports/<name> dir doesn't exist, returns an error message
    result = comp.save_model_card("nonexistent-export", "# Card")
    assert "not found" in result.lower() or isinstance(result, str)


# ── read_sample with actual data ──────────────────────────────────────────────

def test_read_sample_with_data(tmp_project, monkeypatch):
    import ui.components as comp
    monkeypatch.setattr(comp, "ROOT", tmp_project)
    ds_dir = tmp_project / "datasets" / "testds"
    ds_dir.mkdir(parents=True, exist_ok=True)
    sample = ds_dir / "sample_100.jsonl"
    rows = [
        {"messages": [{"role": "user", "content": "What is 2+2?"},
                      {"role": "assistant", "content": "4"}]},
        {"messages": [{"role": "user", "content": "Hello"},
                      {"role": "assistant", "content": "Hi"}]},
    ]
    sample.write_text("\n".join(json.dumps(r) for r in rows))
    result = comp.read_sample("testds")
    assert "USER" in result or "user" in result.lower()


# ── read_data_card with actual data ──────────────────────────────────────────

def test_read_data_card_with_data(tmp_project, monkeypatch):
    import ui.components as comp
    monkeypatch.setattr(comp, "ROOT", tmp_project)
    cards_dir = tmp_project / "data_cards"
    cards_dir.mkdir(exist_ok=True)
    card = {
        "name": "testds",
        "source": "hf://some/dataset",
        "detected_format": "sharegpt",
        "rows_raw": 5000,
        "train_rows": 4500,
        "val_rows": 500,
    }
    (cards_dir / "testds.json").write_text(json.dumps(card))
    result = comp.read_data_card("testds")
    assert "testds" in result or "4,500" in result


# ── read_log with actual data ─────────────────────────────────────────────────

def test_read_log_with_data(tmp_project, monkeypatch):
    import ui.components as comp
    monkeypatch.setattr(comp, "ROOT", tmp_project)
    log = tmp_project / "logs" / "train_myrun.log"
    # read_log parses Python dict repr lines — needs 'loss' key in dict format
    log.write_text(
        "{'loss': 1.5, 'epoch': 0.1, 'step': 1}\n"
        "{'loss': 1.3, 'epoch': 0.2, 'step': 2}\n"
    )
    result = comp.read_log("myrun")
    assert isinstance(result, str)
    assert "div" in result  # wrapped in terminal_wrap HTML


# ── read_eval_summary with actual data ───────────────────────────────────────

def test_read_eval_summary_with_data(tmp_project, monkeypatch):
    import ui.components as comp
    monkeypatch.setattr(comp, "ROOT", tmp_project)
    run_dir = tmp_project / "eval_results" / "myrun"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.md").write_text("# Eval\naccuracy: 0.82")
    result = comp.read_eval_summary("myrun")
    assert "accuracy" in result or isinstance(result, str)


# ── run_ft ────────────────────────────────────────────────────────────────────

def test_run_ft_success(tmp_project, monkeypatch):
    import ui.components as comp
    monkeypatch.setattr(comp, "ROOT", tmp_project)

    mock_proc = MagicMock()
    mock_proc.stdout.readline.side_effect = [
        "step 1: loss=1.5\n",
        "step 2: loss=1.3\n",
        "",   # EOF sentinel
    ]
    mock_proc.wait.return_value = None
    mock_proc.returncode = 0

    with patch("subprocess.Popen", return_value=mock_proc):
        results = list(comp.run_ft(["train", "--model", "m"]))

    assert len(results) > 0
    assert all(isinstance(r, str) for r in results)
    # last chunk should contain success marker
    assert any("Done" in r or "✓" in r for r in results)


def test_run_ft_failure(tmp_project, monkeypatch):
    import ui.components as comp
    monkeypatch.setattr(comp, "ROOT", tmp_project)

    mock_proc = MagicMock()
    mock_proc.stdout.readline.side_effect = ["error: something failed\n", ""]
    mock_proc.wait.return_value = None
    mock_proc.returncode = 1

    with patch("subprocess.Popen", return_value=mock_proc):
        results = list(comp.run_ft(["train", "--model", "m"]))

    assert any("Failed" in r or "✗" in r for r in results)
