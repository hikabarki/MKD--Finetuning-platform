"""Tests for platform/training.py — subprocess wrapper stub."""
from __future__ import annotations

from unittest.mock import MagicMock, patch


def test_run_vl_training_subprocess_yields(tmp_path):
    from ft_platform.training import _run_vl_training_subprocess

    fake_output = [
        "Epoch 1/3\n",
        "step 10: loss=1.2\n",
        "Running eval...\n",
        "Saving model\n",
    ]
    mock_proc = MagicMock()
    mock_proc.stdout = iter(fake_output)
    mock_proc.wait.return_value = 0

    with patch("subprocess.Popen", return_value=mock_proc):
        results = list(_run_vl_training_subprocess(
            cmd=["python", "trainer.py"],
            run_id="test_run",
            output_fn=lambda stage: f"<html>{stage}</html>",
        ))

    assert len(results) > 0
    assert all(isinstance(r, str) for r in results)


def test_run_vl_training_subprocess_no_stdout(tmp_path):
    from ft_platform.training import _run_vl_training_subprocess

    mock_proc = MagicMock()
    mock_proc.stdout = None
    mock_proc.wait.return_value = 0

    with patch("subprocess.Popen", return_value=mock_proc):
        results = list(_run_vl_training_subprocess(
            cmd=["python", "trainer.py"],
            run_id="test_run",
            output_fn=lambda stage: stage,
        ))

    assert results == []
