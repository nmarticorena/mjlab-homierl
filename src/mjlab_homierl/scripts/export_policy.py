"""Regenerate a HOMIE `policy.onnx` + `metadata.yaml` from a run's `.pt` weights.

The checkpoint is resolved from a W&B run:
  1. the run's local log dir (W&B config `log_dir`, or `--log-dir`), since
     HOMIE runners keep checkpoints local (`upload_model=False`);
  2. otherwise the `model_*.pt` files uploaded to the W&B run, if any.
A local `--checkpoint-file` skips W&B entirely.

The env is built from the task's *training* config (1 env) so command ranges
and the rest of the metadata match what `HomieHimOnPolicyRunner.save()` writes
during training.

Usage:
  uv run homie-export --task Mjlab-Homie-Unitree-G1-v2 \\
      --wandb-run-path <entity>/<project>/<run_id> [--checkpoint model_8000.pt]
  uv run homie-export --task Mjlab-Homie-Unitree-G1 \\
      --checkpoint-file logs/rsl_rl/g1_homie_himppo/<run>/model_4000.pt
"""

from __future__ import annotations

import argparse
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

# Fields that must match training for the policy to behave on the robot.
_CONTRACT_KEYS = (
  "joint_names",
  "joint_stiffness",
  "joint_damping",
  "default_joint_pos",
  "action_scale",
  "action_joint_names",
  "one_step_obs_layout",
  "obs_history_length",
)


def _latest_or_named(files: list[str], name: str | None, where: str) -> str:
  ckpts = [f for f in files if re.match(r"^model_\d+\.pt$", f)]
  if not ckpts:
    raise FileNotFoundError(f"No model_*.pt checkpoints in {where}.")
  if name is None:
    return max(ckpts, key=lambda f: int(f.split("_")[1].split(".")[0]))
  if name not in ckpts:
    raise FileNotFoundError(f"'{name}' not in {where}. Available: {sorted(ckpts)}")
  return name


def _resolve_wandb_checkpoint(
  run_path: str, checkpoint: str | None, log_dir: Path | None, download_root: Path
) -> tuple[Path, Any]:
  import wandb

  run = wandb.Api().run(run_path)
  if log_dir is None and run.config.get("log_dir"):
    log_dir = Path(run.config["log_dir"])
  if log_dir is None or not log_dir.is_dir():
    # Fall back to the default log root: rsl_rl names the W&B run after the
    # log dir, so `logs/rsl_rl/<experiment>/<run.name>`.
    matches = list(Path("logs/rsl_rl").glob(f"*/{run.name}"))
    log_dir = matches[0] if matches else None

  if log_dir is not None and log_dir.is_dir():
    files = [p.name for p in log_dir.glob("model_*.pt")]
    if files:
      name = _latest_or_named(files, checkpoint, str(log_dir))
      return log_dir / name, run

  files = [f.name for f in run.files(pattern="model_%.pt")]
  name = _latest_or_named(
    files, checkpoint, f"local log dir ({log_dir}) or W&B run {run_path}"
  )
  download_dir = download_root / run.id
  if not (download_dir / name).exists():
    download_dir.mkdir(parents=True, exist_ok=True)
    run.file(name).download(str(download_dir), replace=True)
  return download_dir / name, run


def _check_against_training_onnx(ckpt_path: Path, onnx_path: Path) -> None:
  """Warn if the regenerated contract differs from the training-time ONNX.

  Metadata comes from the *current* task config; if the task changed since
  training (gains, default pose, ...), the export would silently disagree with
  the weights. The training run's `<run_dir>/<run_dir>.onnx` is the reference.
  """
  import onnx

  ref = ckpt_path.parent / f"{ckpt_path.parent.name}.onnx"
  if not ref.exists():
    print("[INFO] No training-time ONNX next to the checkpoint; skipping check.")
    return
  old = {p.key: p.value for p in onnx.load(str(ref)).metadata_props}
  new = {p.key: p.value for p in onnx.load(str(onnx_path)).metadata_props}
  diffs = [k for k in _CONTRACT_KEYS if k in old and old[k] != new.get(k)]
  if diffs:
    print(
      f"[WARN] Exported metadata differs from {ref.name} in {diffs}. The task "
      "config has likely changed since training; pass the task the run was "
      "trained with (or check out its train_repo_commit) before deploying."
    )


def _wandb_train_commit(run: Any) -> str | None:
  try:
    return run.metadata["git"]["commit"]
  except Exception:
    return None


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--task", required=True, help="Registered HOMIE task ID.")
  src = parser.add_mutually_exclusive_group(required=True)
  src.add_argument("--wandb-run-path", help="<entity>/<project>/<run_id>")
  src.add_argument("--checkpoint-file", type=Path, help="Local model_*.pt")
  parser.add_argument(
    "--checkpoint",
    help="Checkpoint name in the run, e.g. model_8000.pt (default: latest).",
  )
  parser.add_argument(
    "--log-dir", type=Path, help="Local run dir holding the .pt files (overrides W&B)."
  )
  parser.add_argument(
    "--out-dir",
    type=Path,
    help="Output dir (default: <checkpoint dir>/exported/<checkpoint stem>).",
  )
  parser.add_argument(
    "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
  )
  parser.add_argument(
    "--upload",
    action="store_true",
    help="Also upload policy.onnx + metadata.yaml to the W&B run.",
  )
  args = parser.parse_args()

  # Deferred: these pull in mjlab/warp, keep `--help` fast.
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  import mjlab_homierl  # noqa: F401  (registers tasks)
  from mjlab_homierl.rl.exporter import export_homie_deployment

  run: Any = None
  if args.checkpoint_file is not None:
    ckpt_path = args.checkpoint_file.resolve()
    run_name = ckpt_path.parent.name
  else:
    ckpt_path, run = _resolve_wandb_checkpoint(
      args.wandb_run_path,
      args.checkpoint,
      args.log_dir,
      Path("logs/rsl_rl/wandb_checkpoints"),
    )
    run_name = run.name
  print(f"[INFO] Checkpoint: {ckpt_path}")

  env_cfg = load_env_cfg(args.task)
  env_cfg.scene.num_envs = 1
  agent_cfg = load_rl_cfg(args.task)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
  vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(args.task)
  assert runner_cls is not None
  runner = runner_cls(vec_env, asdict(agent_cfg), log_dir=None, device=args.device)
  runner.load(str(ckpt_path), map_location=args.device)

  extra: dict = {"checkpoint": ckpt_path.name}
  if run is not None:
    extra["wandb_run_path"] = "/".join(run.path)
    commit = _wandb_train_commit(run)
    if commit:
      extra["train_repo_commit"] = commit

  out_dir = args.out_dir or ckpt_path.parent / "exported" / ckpt_path.stem
  onnx_path, yaml_path = export_homie_deployment(
    runner.alg.get_policy(), env, out_dir, run_name, extra_metadata=extra
  )
  print(f"[INFO] Wrote {onnx_path}\n[INFO] Wrote {yaml_path}")
  _check_against_training_onnx(ckpt_path, onnx_path)

  if args.upload:
    if run is None:
      raise SystemExit("--upload requires --wandb-run-path.")
    for file in (onnx_path, yaml_path):
      run.upload_file(str(file), root=str(out_dir))
    print(f"[INFO] Uploaded to {run.url}")

  vec_env.close()


if __name__ == "__main__":
  main()
