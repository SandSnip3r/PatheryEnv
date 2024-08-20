from gymnasium.envs.registration import register

register(
    id="pathery_env/Pathery-v0",
    entry_point="pathery_env.envs:PatheryEnv",
)
