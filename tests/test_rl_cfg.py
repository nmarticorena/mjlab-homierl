from mjlab_homierl.rl_cfg import (
  unitree_g1_homie_himppo_runner_cfg,
  unitree_g1_homie_smooth_himppo_runner_cfg,
  unitree_g1_homie_v2_himppo_runner_cfg,
  unitree_g1_homie_v3_himppo_runner_cfg,
  unitree_h1_homie_himppo_runner_cfg,
)


def test_himppo_runner_cfg_uses_custom_algorithm() -> None:
  for cfg in (
    unitree_g1_homie_himppo_runner_cfg(),
    unitree_g1_homie_smooth_himppo_runner_cfg(),
    unitree_h1_homie_himppo_runner_cfg(),
  ):
    assert cfg.algorithm.class_name == "mjlab_homierl.rl.himppo.algorithm.HIMPPO"
    # OpenHomie HIMActorCritic hidden dims.
    assert cfg.actor.hidden_dims == (512, 256, 256)
    assert cfg.critic.hidden_dims == (512, 256, 256)
    # Checkpoints must never be uploaded to W&B.
    assert cfg.upload_model is False


def test_g1_smooth_runner_cfg() -> None:
  base = unitree_g1_homie_himppo_runner_cfg()
  smooth = unitree_g1_homie_smooth_himppo_runner_cfg()
  assert smooth.algorithm.entropy_coef == base.algorithm.entropy_coef
  assert smooth.max_iterations == 8_000
  assert smooth.experiment_name == base.experiment_name
  assert base.algorithm.entropy_coef == 0.01


def test_g1_v2_runner_cfg() -> None:
  v2 = unitree_g1_homie_v2_himppo_runner_cfg()
  assert v2.experiment_name == "g1_homie_v2_himppo"
  assert v2.algorithm.entropy_coef == 0.01


def test_g1_v3_runner_cfg() -> None:
  v2 = unitree_g1_homie_v2_himppo_runner_cfg()
  v3 = unitree_g1_homie_v3_himppo_runner_cfg()
  assert v3.experiment_name == "g1_homie_v3_himppo"
  assert v3.max_iterations == 4_000
  # Same network as v2, so the v2 checkpoint loads.
  assert v3.actor == v2.actor and v3.critic == v2.critic
  assert v3.upload_model is False
