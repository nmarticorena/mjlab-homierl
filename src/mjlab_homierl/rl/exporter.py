import copy
import importlib.metadata
import json
import os
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)


class _HimOnnxPolicyExporter(torch.nn.Module):
  """ONNX wrapper for HOMIE HIMActorCritic policies.

  The standard Isaac-Lab-style exporter only serializes `policy.actor`, but
  HIMActorCritic requires estimator-based preprocessing from the raw actor
  observation history. This wrapper exports the full inference path:

    obs_history -> normalizer -> HIMActorCritic.act_inference_actor_obs -> actions
  """

  def __init__(self, actor_critic: object, normalizer: object | None, verbose: bool):
    super().__init__()
    self.verbose = bool(verbose)

    # NOTE: Do not deepcopy the full actor_critic module.
    # HIMPPO policies cache non-leaf tensors (e.g., action distribution mean/std)
    # which PyTorch does not allow to be deep-copied in newer versions.
    self.actor = copy.deepcopy(actor_critic.actor)
    self.estimator = copy.deepcopy(actor_critic.estimator.encoder)
    self.num_actor_obs = int(actor_critic.num_actor_obs)
    self.num_one_step_obs = int(actor_critic.num_one_step_obs)
    self.actor_proprioceptive_obs_length = int(
      actor_critic.actor_proprioceptive_obs_length
    )
    self.num_height_points = int(getattr(actor_critic, "num_height_points", 0))
    self.actor_use_height = bool(getattr(actor_critic, "actor_use_height", False))
    if self.actor_use_height:
      self.terrain_encoder = copy.deepcopy(actor_critic.terrain_encoder)
    else:
      self.terrain_encoder = None

    self.actor.eval()
    self.estimator.eval()
    if self.terrain_encoder is not None:
      self.terrain_encoder.eval()

    if normalizer is not None:
      self.normalizer = copy.deepcopy(normalizer)
    else:
      self.normalizer = torch.nn.Identity()

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    obs = self.normalizer(obs)

    parts = self.estimator(obs[:, : self.actor_proprioceptive_obs_length])
    vel, z = parts[..., :3], parts[..., 3:]
    z = F.normalize(z, dim=-1, p=2.0)

    if self.actor_use_height:
      assert self.terrain_encoder is not None
      terrain_in = obs[:, -(self.num_height_points + self.num_one_step_obs) :]
      terrain_latent = self.terrain_encoder(terrain_in)
      last_step = obs[
        :, -(self.num_height_points + self.num_one_step_obs) : -self.num_height_points
      ]
      actor_in = torch.cat((last_step, vel, z, terrain_latent), dim=-1)
    else:
      last_step = obs[:, -self.num_one_step_obs :]
      actor_in = torch.cat((last_step, vel, z), dim=-1)

    return self.actor(actor_in)

  def export(self, path: str, filename: str) -> None:
    self.to("cpu")
    self.eval()

    obs = torch.zeros(1, self.num_actor_obs, dtype=torch.float32)
    torch.onnx.export(
      self,
      obs,
      os.path.join(path, filename),
      export_params=True,
      opset_version=11,
      verbose=self.verbose,
      input_names=["obs"],
      output_names=["actions"],
      dynamic_axes={},
      dynamo=False,
    )


def export_homie_policy_as_onnx(
  actor_critic: object,
  path: str,
  normalizer: object | None = None,
  filename="policy.onnx",
  verbose=False,
):
  if not os.path.exists(path):
    os.makedirs(path, exist_ok=True)

  if not (
    hasattr(actor_critic, "act_inference_actor_obs")
    and hasattr(actor_critic, "estimator")
  ):
    raise TypeError("HOMIE ONNX export expects a HIMActorCritic-compatible policy.")

  policy_exporter = _HimOnnxPolicyExporter(actor_critic, normalizer, verbose)
  policy_exporter.export(path, filename)


def _train_repo_commit() -> str:
  """Git commit of this training repo at export time (provenance)."""
  try:
    return (
      subprocess.run(
        ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        timeout=5.0,
        check=True,
      ).stdout.strip()
      or "unknown"
    )
  except Exception:
    return "unknown"


def homie_extra_metadata(env: ManagerBasedRlEnv) -> dict:
  """HOMIE-specific ONNX metadata beyond mjlab's base fields.

  Everything a downstream controller plugin needs to run the policy without
  hardcoding conventions: the paired init keyframe base height, the height
  command semantics, the observation scaling/layout, and training provenance.
  """
  metadata: dict = {"train_repo_commit": _train_repo_commit()}

  # Physics-stack provenance: the benchmark runtime must match these versions
  # or settle behavior and gait drift relative to training.
  for pkg in ("mjlab", "mujoco", "mujoco-warp"):
    try:
      metadata[f"version_{pkg.replace('-', '_')}"] = importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
      pass

  # Init keyframe: joint pose (base metadata `default_joint_pos`) and base
  # height must travel as a pair.
  robot_cfg = env.cfg.scene.entities["robot"]
  init_state = robot_cfg.init_state
  if init_state is not None and init_state.pos is not None:
    metadata["init_base_height"] = float(init_state.pos[2])

  # Height command semantics (relative pelvis height above the lowest foot).
  try:
    height_cfg = env.command_manager.get_term("height").cfg
    metadata["height_command_range"] = list(height_cfg.ranges.height)
    metadata["standing_height"] = float(height_cfg.standing_height)
  except KeyError:
    pass

  # Twist command semantics: body(base)-frame velocities and training ranges.
  # Commands outside these ranges are out-of-distribution for the policy.
  try:
    twist_ranges = env.command_manager.get_term("twist").cfg.ranges
    metadata["command_frame"] = "base"
    metadata["twist_command_ranges"] = {
      "lin_vel_x": list(twist_ranges.lin_vel_x),
      "lin_vel_y": list(twist_ranges.lin_vel_y),
      "ang_vel_z": list(twist_ranges.ang_vel_z),
    }
  except KeyError:
    pass

  # The joints the policy's action vector maps to, in output order (the base
  # `joint_names` field lists the full robot; actions cover a subset).
  action_term = env.action_manager.get_term("joint_pos")
  metadata["action_joint_names"] = list(action_term._target_names)

  # Observation scaling and layout of the flattened actor history.
  obs_term_cfg = env.observation_manager.get_term_cfg("actor", "him_obs")
  obs_scales = obs_term_cfg.params.get("obs_scales")
  if obs_scales is not None:
    for key, value in obs_scales.items():
      metadata[f"obs_scale_{key}"] = float(value)
  history_length = max(1, int(obs_term_cfg.history_length))
  metadata["obs_history_length"] = history_length
  actor_dim = env.observation_manager.group_obs_dim["actor"][0]
  metadata["num_one_step_obs"] = int(actor_dim) // history_length

  # Field order of one step of the actor observation (see
  # mdp.observations.him_actor_one_step_obs). joint_pos/joint_vel follow the
  # full `joint_names` order; commands are (vx, vy, wz, height).
  robot = env.scene["robot"]
  num_joints = len(robot.joint_names)
  metadata["one_step_obs_layout"] = {
    "command": 4,
    "base_ang_vel": 3,
    "projected_gravity": 3,
    "joint_pos_rel": num_joints,
    "joint_vel": num_joints,
    "last_action": int(action_term.action_dim),
  }

  # Low-level control: effort limits (natural joint order, like the base
  # stiffness/damping) and the control rate the policy was trained at.
  joint_name_to_ctrl_id = {
    actuator.target.split("/")[-1]: actuator.id for actuator in robot.spec.actuators
  }
  ctrl_ids_natural = [
    joint_name_to_ctrl_id[name]
    for name in robot.joint_names
    if name in joint_name_to_ctrl_id
  ]
  metadata["joint_effort_limit"] = env.sim.mj_model.actuator_forcerange[
    ctrl_ids_natural, 1
  ].tolist()
  physics_timestep = float(env.sim.mj_model.opt.timestep)
  decimation = int(env.cfg.decimation)
  metadata["physics_timestep_s"] = physics_timestep
  metadata["control_decimation"] = decimation
  metadata["control_dt_s"] = physics_timestep * decimation
  metadata["control_rate_hz"] = 1.0 / (physics_timestep * decimation)

  return metadata


def build_homie_metadata(
  env: ManagerBasedRlEnv, run_path: str, extra: dict | None = None
) -> dict:
  """Base mjlab metadata + HOMIE fields (+ `extra` overrides), native types.

  Single source for both the ONNX metadata props and `metadata.yaml`, so the
  two never disagree.
  """
  metadata = get_base_metadata(env, run_path)
  metadata.update(homie_extra_metadata(env))
  metadata.update(extra or {})
  return metadata


def write_metadata_yaml(metadata: dict, path: str | Path) -> Path:
  """Write `metadata` as a human-readable YAML next to the ONNX export."""
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w") as f:
    yaml.safe_dump(metadata, f, sort_keys=False)
  return path


def attach_onnx_metadata(
  env: ManagerBasedRlEnv,
  run_path: str,
  path: str,
  filename="policy.onnx",
  metadata: dict | None = None,
) -> None:
  """Attach base + HOMIE-specific metadata to an exported ONNX model.

  ONNX metadata props are strings: lists become CSV (mjlab convention) and
  dicts become JSON, which is what `mjlab_homierl.runtime` parses.

  Args:
    env: The RL environment.
    run_path: W&B run path or other identifier.
    path: Directory containing the ONNX file.
    filename: Name of the ONNX file.
    metadata: Prebuilt metadata (see `build_homie_metadata`); built if None.
  """
  onnx_path = os.path.join(path, filename)
  if metadata is None:
    metadata = build_homie_metadata(env, run_path)
  attach_metadata_to_onnx(
    onnx_path,
    {k: json.dumps(v) if isinstance(v, dict) else v for k, v in metadata.items()},
  )


def export_homie_deployment(
  actor_critic: object,
  env: ManagerBasedRlEnv,
  out_dir: str | Path,
  run_path: str,
  onnx_filename: str = "policy.onnx",
  yaml_filename: str = "metadata.yaml",
  extra_metadata: dict | None = None,
) -> tuple[Path, Path]:
  """Export the policy ONNX (with metadata props) and `metadata.yaml`.

  Returns:
    (onnx_path, yaml_path)
  """
  out_dir = Path(out_dir)
  export_homie_policy_as_onnx(
    actor_critic, path=str(out_dir), normalizer=None, filename=onnx_filename
  )
  metadata = build_homie_metadata(env, run_path, extra_metadata)
  attach_onnx_metadata(
    env, run_path, path=str(out_dir), filename=onnx_filename, metadata=metadata
  )
  yaml_path = write_metadata_yaml(metadata, out_dir / yaml_filename)
  return out_dir / onnx_filename, yaml_path
