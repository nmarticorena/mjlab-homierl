"""Checkpoint resolution for the `homie-export` W&B regeneration script."""

import sys
import types
from pathlib import Path

import pytest

from mjlab_homierl.scripts import export_policy


class _FakeFile:
  def __init__(self, name: str) -> None:
    self.name = name

  def download(self, root: str, replace: bool = False) -> None:
    (Path(root) / self.name).write_bytes(b"")


class _FakeRun:
  def __init__(self, log_dir: str | None, remote: list[str]) -> None:
    self.config = {"log_dir": log_dir} if log_dir else {}
    self.name = "2026-01-01_00-00-00_run"
    self.id = "abc123"
    self._remote = remote

  def files(self, pattern: str | None = None) -> list[_FakeFile]:
    return [_FakeFile(n) for n in self._remote]

  def file(self, name: str) -> _FakeFile:
    return _FakeFile(name)


def _patch_wandb(monkeypatch: pytest.MonkeyPatch, run: _FakeRun) -> None:
  fake = types.SimpleNamespace(Api=lambda: types.SimpleNamespace(run=lambda _: run))
  monkeypatch.setitem(sys.modules, "wandb", fake)


def test_latest_or_named() -> None:
  files = ["model_200.pt", "model_1000.pt", "model_50.pt", "policy.onnx"]
  assert export_policy._latest_or_named(files, None, "x") == "model_1000.pt"
  assert export_policy._latest_or_named(files, "model_50.pt", "x") == "model_50.pt"
  with pytest.raises(FileNotFoundError):
    export_policy._latest_or_named(files, "model_7.pt", "x")
  with pytest.raises(FileNotFoundError):
    export_policy._latest_or_named(["policy.onnx"], None, "x")


def test_resolves_local_log_dir_from_wandb_config(
  monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
  for it in (0, 200, 400):
    (tmp_path / f"model_{it}.pt").write_bytes(b"")
  _patch_wandb(monkeypatch, _FakeRun(str(tmp_path), remote=[]))
  path, _ = export_policy._resolve_wandb_checkpoint(
    "e/p/abc123", None, None, tmp_path / "dl"
  )
  assert path == tmp_path / "model_400.pt"


def test_falls_back_to_wandb_files(
  monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
  monkeypatch.chdir(tmp_path)
  run = _FakeRun("/nonexistent/run", remote=["model_100.pt", "model_300.pt"])
  _patch_wandb(monkeypatch, run)
  path, _ = export_policy._resolve_wandb_checkpoint(
    "e/p/abc123", None, None, tmp_path / "dl"
  )
  assert path == tmp_path / "dl" / "abc123" / "model_300.pt"
  assert path.exists()
