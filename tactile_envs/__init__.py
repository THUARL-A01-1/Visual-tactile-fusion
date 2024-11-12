from gymnasium.envs.registration import register

print("Registering Tactile Envs")
register(
     id="tactile_envs/Regrasp-v0",
     entry_point="tactile_envs.envs:RegraspEnv",
     max_episode_steps=300,
)