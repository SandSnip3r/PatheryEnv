from gymnasium.envs.registration import register

register(
  id="pathery_env/Pathery-HardCoded",
  entry_point="pathery_env.envs:createHardCoded",
)

register(
  id="pathery_env/Pathery-RandomNormal",
  entry_point="pathery_env.envs:createRandomNormal",
)

register(
  id="pathery_env/Pathery-FromMapString",
  entry_point="pathery_env.envs:fromMapString",
)
