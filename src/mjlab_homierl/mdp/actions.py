"""Policy-free action terms for HOMIE upper-body and gripper motion."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

import torch
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class DelayedJointPositionActionCfg(JointPositionActionCfg):
  """Joint position action with randomized actuation latency.

  OpenHomie DR (``domain_rand.delay``): each policy step, every env draws a
  delay of 0..max physics substeps; the previous position target is held for
  that many substeps before the new one takes effect. This models the real
  robot's control-loop/communication latency (one policy step spans
  ``decimation`` physics substeps, 5 ms each on the HOMIE tasks).
  """

  max_delay_substeps: int | None = None
  """Maximum delay in physics substeps. ``None`` uses ``decimation - 1``
  (OpenHomie's choice); ``0`` disables the delay entirely (play/eval)."""

  def build(self, env: "ManagerBasedRlEnv") -> "DelayedJointPositionAction":
    return DelayedJointPositionAction(self, env)


class DelayedJointPositionAction(JointPositionAction):
  """Applies the previous target for the first ``delay`` substeps of a step."""

  cfg: "DelayedJointPositionActionCfg"

  def __init__(self, cfg: "DelayedJointPositionActionCfg", env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)
    self._max_delay = (
      env.cfg.decimation - 1
      if cfg.max_delay_substeps is None
      else int(cfg.max_delay_substeps)
    )
    if not 0 <= self._max_delay < env.cfg.decimation:
      raise ValueError(
        f"max_delay_substeps must be in [0, {env.cfg.decimation - 1}], got"
        f" {self._max_delay}."
      )
    # Holding the default pose is the correct "previous target" after a reset.
    self._default_target = self._entity.data.default_joint_pos[
      :, self._target_ids
    ].clone()
    self._prev_target = self._default_target.clone()
    self._delay = torch.zeros(self.num_envs, 1, device=self.device)
    self._substep = 0

  def process_actions(self, actions: torch.Tensor) -> None:
    self._prev_target = self._processed_actions.clone()
    super().process_actions(actions)
    if self._max_delay > 0:
      self._delay = torch.randint(
        0, self._max_delay + 1, (self.num_envs, 1), device=self.device
      ).float()
    self._substep = 0

  def apply_actions(self) -> None:
    if self._max_delay == 0:
      super().apply_actions()
      return
    use_new = (self._substep >= self._delay).float()
    target = self._prev_target + (self._processed_actions - self._prev_target) * use_new
    encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
    self._entity.set_joint_position_target(
      target - encoder_bias, joint_ids=self._target_ids
    )
    self._substep += 1

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    if env_ids is None:
      env_ids = slice(None)
    self._prev_target[env_ids] = self._default_target[env_ids]
    # Freshly reset envs also have stale processed actions; align them so the
    # first delayed substeps hold the default pose rather than a dead target.
    self._processed_actions[env_ids] = self._default_target[env_ids]


class UpperBodyPoseAction(ActionTerm):
  """Curriculum-driven random pose targets for the upper body.

  Contributes zero policy dimensions. Goals are resampled by an interval event
  (:func:`sample_upper_body_goals`) and reached by linear interpolation over the
  resampling interval, reproducing OpenHomie's upper-body disturbance scheme:

  - A global curriculum ratio ``r`` grows as locomotion improves.
  - Per (env, joint), the target magnitude is drawn from a truncated-exponential
    transform of ``r`` (heavily biased toward small motions early on), times a
    uniform factor.
  - The target direction is a fair coin between the joint's lower and upper hard
    limit, so target amplitudes are proportional to each joint's range.

  Optionally (``reach_probability > 0``), environments also enter "hands
  forward" reach scenarios, mixed with the random goals: the arms move to a
  reach pose over ``reach_transition_s`` and hold it for ``reach_hold_s``, like
  a deployed arm IK raising the hands to a handle. The pose is a configured
  anchor plus noise, or a uniform sample from ``reach_box``.

  Likewise (``static_probability > 0``), environments enter static scenarios:
  the arms stop where they are and stay still for ``static_hold_s``, like a
  deployed arm holding its pose while the robot stands or walks.
  """

  cfg: "UpperBodyPoseActionCfg"

  def __init__(self, cfg: "UpperBodyPoseActionCfg", env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)

    joint_ids, joint_names = self._entity.find_joints(
      cfg.joint_names, preserve_order=True
    )
    if len(joint_ids) == 0:
      raise ValueError(f"No upper-body joints matched patterns: {cfg.joint_names}.")

    self._joint_ids = torch.tensor(joint_ids, device=self.device, dtype=torch.long)
    self._joint_names = joint_names
    self._raw_actions = torch.zeros(self.num_envs, 0, device=self.device)

    self._default = self._entity.data.default_joint_pos[:, self._joint_ids].clone()
    limits = self._entity.data.joint_pos_limits[:, self._joint_ids]
    self._lower_offset = limits[..., 0] - self._default
    self._upper_offset = limits[..., 1] - self._default

    self._current = self._default.clone()
    self._goal = self._default.clone()
    self._delta = torch.zeros_like(self._default)
    self._interval_steps = max(1, int(round(cfg.interval_s / env.step_dt)))

    # Reach and static scenarios. _scenario_left counts the goal resamplings
    # left in an env's current scenario (0: random goals); _in_reach tells a
    # reach from a static hold.
    self._lower = limits[..., 0]
    self._upper = limits[..., 1]
    self._scenario_left = torch.zeros(
      self.num_envs, device=self.device, dtype=torch.long
    )
    self._in_reach = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
    self._reach_transition_steps = max(
      1, int(round(cfg.reach_transition_s / env.step_dt))
    )
    unknown = {
      name
      for pose in (*cfg.reach_anchors, cfg.reach_box)
      for name in pose
      if name not in joint_names
    }
    if unknown:
      raise ValueError(f"Reach poses name non-upper-body joints: {sorted(unknown)}.")
    default = self._default[0]
    self._reach_anchors = (
      torch.stack(
        [
          torch.tensor(
            [anchor.get(name, float(default[i])) for i, name in enumerate(joint_names)],
            device=self.device,
          )
          for anchor in cfg.reach_anchors
        ]
      )
      if cfg.reach_anchors
      else torch.zeros(0, len(joint_names), device=self.device)
    )
    self._reach_box_low = torch.tensor(
      [
        cfg.reach_box.get(name, (float(default[i]),) * 2)[0]
        for i, name in enumerate(joint_names)
      ],
      device=self.device,
    )
    self._reach_box_high = torch.tensor(
      [
        cfg.reach_box.get(name, (float(default[i]),) * 2)[1]
        for i, name in enumerate(joint_names)
      ],
      device=self.device,
    )

    self._curriculum_ratio = torch.tensor(
      cfg.initial_ratio, device=self.device, dtype=torch.float32
    )

  # ActionTerm interface.

  @property
  def action_dim(self) -> int:
    return 0

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  def process_actions(self, actions: torch.Tensor) -> None:
    if actions.numel() != 0:
      raise ValueError(
        f"UpperBodyPoseAction expects zero-dim actions, got shape {actions.shape}."
      )
    # Advance the interpolation once per control step, without overshooting.
    # The delta always points toward the goal (set at sampling time), so a
    # magnitude clamp against the remaining distance is sufficient.
    remaining = self._goal - self._current
    step = torch.clamp(self._delta, min=-remaining.abs(), max=remaining.abs())
    self._current = self._current + step

  def apply_actions(self) -> None:
    self._entity.set_joint_position_target(self._current, joint_ids=self._joint_ids)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._current[env_ids] = self._default[env_ids]
    self._goal[env_ids] = self._default[env_ids]
    self._delta[env_ids] = 0.0
    self._scenario_left[env_ids] = 0
    self._in_reach[env_ids] = False

  # HOMIE goal sampling and curriculum.

  def sample_new_goals(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None or isinstance(env_ids, slice):
      env_ids = torch.arange(self.num_envs, device=self.device)[
        slice(None) if env_ids is None else env_ids
      ]
    random_ids = env_ids
    if self.cfg.reach_probability > 0.0 or self.cfg.static_probability > 0.0:
      # Envs mid-scenario keep their goal; the others may start a reach or a
      # static hold, with probabilities that grow with the curriculum ratio.
      left = self._scenario_left[env_ids]
      holding = left > 1
      self._scenario_left[env_ids] = torch.clamp(left - 1, min=0)
      self._in_reach[env_ids] &= holding
      ratio = float(self._curriculum_ratio.clamp(0.0, 1.0))
      reach_p = self.cfg.reach_probability * ratio
      static_p = self.cfg.static_probability * ratio
      u = torch.rand(len(env_ids), device=self.device)
      start_reach = ~holding & (u < reach_p)
      start_static = ~holding & ~start_reach & (u < reach_p + static_p)
      self._start_reach(env_ids[start_reach])
      self._start_static(env_ids[start_static])
      random_ids = env_ids[~holding & ~start_reach & ~start_static]
    if len(random_ids) > 0:
      self._sample_random_goals(random_ids)

  def _start_reach(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    if n == 0:
      return
    num_joints = self._default.shape[1]
    box = self._reach_box_low + torch.rand(n, num_joints, device=self.device) * (
      self._reach_box_high - self._reach_box_low
    )
    pose = box
    if len(self._reach_anchors) > 0:
      anchor = self._reach_anchors[
        torch.randint(len(self._reach_anchors), (n,), device=self.device)
      ]
      noise = self.cfg.reach_anchor_noise * (
        2.0 * torch.rand(n, num_joints, device=self.device) - 1.0
      )
      use_anchor = (
        torch.rand(n, 1, device=self.device) < self.cfg.reach_anchor_probability
      )
      pose = torch.where(use_anchor, anchor + noise, box)
    goal = torch.clamp(pose, self._lower[env_ids], self._upper[env_ids])
    self._goal[env_ids] = goal
    self._delta[env_ids] = (goal - self._current[env_ids]) / float(
      self._reach_transition_steps
    )
    self._scenario_left[env_ids] = self._sample_intervals(n, self.cfg.reach_hold_s)
    self._in_reach[env_ids] = True

  def _start_static(self, env_ids: torch.Tensor) -> None:
    """Stop the arms where they are and hold them still."""
    n = len(env_ids)
    if n == 0:
      return
    self._goal[env_ids] = self._current[env_ids]
    self._delta[env_ids] = 0.0
    self._scenario_left[env_ids] = self._sample_intervals(n, self.cfg.static_hold_s)

  def _sample_intervals(self, n: int, duration_s: tuple[float, float]) -> torch.Tensor:
    low, high = duration_s
    seconds = low + torch.rand(n, device=self.device) * (high - low)
    return torch.clamp(torch.round(seconds / self.cfg.interval_s).long(), min=1)

  @property
  def in_reach(self) -> torch.Tensor:
    """Per env: whether a reach scenario is active."""
    return self._in_reach & (self._scenario_left > 0)

  @property
  def in_static(self) -> torch.Tensor:
    """Per env: whether a static hold is active."""
    return ~self._in_reach & (self._scenario_left > 0)

  def _sample_random_goals(self, env_ids: torch.Tensor) -> None:
    defaults = self._default[env_ids]
    shape = defaults.shape

    # Truncated-exponential ratio transform (OpenHomie): for small curriculum
    # ratios almost all samples are near zero; as the ratio approaches 1 the
    # distribution flattens toward uniform.
    r = float(self._curriculum_ratio.clamp(0.0, 1.0).item())
    k = 20.0 * (1.0 - 0.99 * r)
    u = torch.rand(shape, device=self.device)
    ratio = -1.0 / k * torch.log(1.0 - u + u * math.exp(-k))
    ratio = ratio * torch.rand(shape, device=self.device)

    side = torch.rand(shape, device=self.device) < 0.5
    offset = torch.where(side, self._lower_offset[env_ids], self._upper_offset[env_ids])
    self._goal[env_ids] = defaults + ratio * offset

    # Reach the goal by linear interpolation over one resampling interval.
    self._delta[env_ids] = (self._goal[env_ids] - self._current[env_ids]) / float(
      self._interval_steps
    )

  def set_curriculum_ratio(self, ratio: torch.Tensor | float) -> None:
    ratio_tensor = torch.as_tensor(ratio, device=self.device, dtype=torch.float32)
    self._curriculum_ratio = ratio_tensor.clamp(0.0, 1.0)

  @property
  def curriculum_ratio(self) -> torch.Tensor:
    return self._curriculum_ratio


@dataclass(kw_only=True)
class UpperBodyPoseActionCfg(ActionTermCfg):
  """Configuration for :class:`UpperBodyPoseAction`."""

  joint_names: Sequence[str]
  interval_s: float = 1.0
  """Goal resampling interval; must match the driving interval event."""
  initial_ratio: float = 0.0
  """Initial curriculum ratio (0 disables upper-body motion)."""
  reach_probability: float = 0.0
  """Per env and goal resampling, the probability (times the curriculum
  ratio) that an env not in a reach scenario starts one. 0 keeps OpenHomie's
  random goals only."""
  reach_hold_s: tuple[float, float] = (3.0, 8.0)
  """Duration range of a reach scenario, transition included."""
  reach_transition_s: float = 2.0
  """Time to move from the current pose to the reach pose."""
  reach_anchors: tuple[dict[str, float], ...] = ()
  """Reach poses by joint name; unlisted joints keep their default."""
  reach_anchor_probability: float = 0.5
  """Probability that a reach uses an anchor (plus noise) instead of
  ``reach_box``."""
  reach_anchor_noise: float = 0.2
  """Uniform per-joint noise (rad) added to an anchor."""
  reach_box: dict[str, tuple[float, float]] = field(default_factory=dict)
  """Per-joint (low, high) ranges for box-sampled reach poses; unlisted joints
  keep their default."""
  static_probability: float = 0.0
  """Per env and goal resampling, the probability (times the curriculum
  ratio) that an env not in a scenario stops its arms where they are and holds
  them still. 0 disables static holds."""
  static_hold_s: tuple[float, float] = (2.0, 6.0)
  """Duration range of a static hold."""

  def build(self, env: ManagerBasedRlEnv) -> UpperBodyPoseAction:
    return UpperBodyPoseAction(self, env)


def sample_upper_body_goals(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  action_name: str,
  start_step: int = 0,
) -> None:
  """Interval event: resample upper-body pose goals for all environments."""
  del env_ids  # Sampling is global, as in OpenHomie.
  if env.common_step_counter < start_step:
    return
  term = env.action_manager.get_term(action_name)
  if not isinstance(term, UpperBodyPoseAction):
    raise TypeError(
      f"Action term '{action_name}' must be UpperBodyPoseAction, got {type(term)}."
    )
  term.sample_new_goals()


class GripperActuatorAction(ActionTerm):
  """Drive XML-defined gripper actuators with internal random targets."""

  cfg: "GripperActuatorActionCfg"

  def __init__(self, cfg: "GripperActuatorActionCfg", env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)
    act_ids, act_names = self._entity.find_actuators(
      cfg.actuator_names, preserve_order=True
    )
    if len(act_ids) == 0:
      raise ValueError(
        f"No actuators matched for gripper action: {cfg.actuator_names}."
      )
    self._actuator_ids = torch.tensor(act_ids, device=self.device, dtype=torch.long)
    self._actuator_names = act_names
    self._raw_actions = torch.zeros(self.num_envs, 0, device=self.device)

    ctrl_mins, ctrl_maxs = [], []
    for idx in act_ids:
      ctrlrange = self._entity.spec.actuators[idx].ctrlrange
      lo, hi = (-1.0, 1.0) if ctrlrange is None else (ctrlrange[0], ctrlrange[1])
      ctrl_mins.append(lo)
      ctrl_maxs.append(hi)
    self._ctrl_min = torch.tensor(
      ctrl_mins, device=self.device, dtype=torch.float32
    ).unsqueeze(0)
    self._ctrl_max = torch.tensor(
      ctrl_maxs, device=self.device, dtype=torch.float32
    ).unsqueeze(0)

    self._current = torch.zeros(self.num_envs, len(act_ids), device=self.device)
    self._goal = torch.zeros_like(self._current)

  @property
  def action_dim(self) -> int:
    return 0

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  def process_actions(self, actions: torch.Tensor) -> None:
    if actions.numel() != 0:
      raise ValueError(
        f"GripperActuatorAction expects zero-dim actions, got shape {actions.shape}."
      )
    self._current = torch.lerp(self._current, self._goal, self.cfg.interp_rate)

  def apply_actions(self) -> None:
    ctrl = self._ctrl_min + (self._ctrl_max - self._ctrl_min) * self._current
    self._entity.write_ctrl_to_sim(ctrl, ctrl_ids=self._actuator_ids)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._current[env_ids] = 0.0
    self._goal[env_ids] = 0.0

  def sample_new_goals(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    low, high = self.cfg.target_range
    noise = torch.empty_like(self._goal[env_ids]).uniform_(low, high)
    self._goal[env_ids] = torch.clamp(noise, 0.0, 1.0)


@dataclass(kw_only=True)
class GripperActuatorActionCfg(ActionTermCfg):
  actuator_names: Sequence[str]
  target_range: tuple[float, float] = (0.0, 1.0)
  interp_rate: float = 0.05

  def build(self, env: ManagerBasedRlEnv) -> GripperActuatorAction:
    return GripperActuatorAction(self, env)


def sample_gripper_goals(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  action_name: str,
  start_step: int = 0,
) -> None:
  """Interval event: resample gripper targets."""
  if env.common_step_counter < start_step:
    return
  term = env.action_manager.get_term(action_name)
  if not isinstance(term, GripperActuatorAction):
    raise TypeError(
      f"Action term '{action_name}' must be GripperActuatorAction, got {type(term)}."
    )
  term.sample_new_goals(env_ids)
