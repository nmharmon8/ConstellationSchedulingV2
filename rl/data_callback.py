from bsk_rl.utils.rllib import EpisodeDataCallbacks

class CustomDataCallbacks(EpisodeDataCallbacks):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def pull_env_metrics(self, env):
        reward = env.cum_reward
        print(f"Env reward: {reward}")

        data = dict(
            reward=reward
        )
        return data