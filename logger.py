from stable_baselines3.common.callbacks import BaseCallback
import os
import numpy as np



class ReturnLogger(BaseCallback):
    def __init__(self, save_dir, env_name, algo_name, policy_id, iter, seed, verbose=0):
        super(ReturnLogger, self).__init__(verbose)
        self.episode_vec_returns = []
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.env_name = env_name
        self.algo_name = algo_name
        self.seed = seed
        self.iter = iter
        self.policy_id = policy_id

    def _on_step(self) -> bool:
        # vec env
        if isinstance(self.locals.get("infos"), tuple) or isinstance(self.locals.get("infos"), list):
            for info in self.locals.get("infos"):
                if "episode" in info:
                    self.episode_vec_returns.append(info["episode"]["r"])
                    if len(self.episode_vec_returns) % 100 == 0:
                        print('return', info["episode"]["r"])
        else:
            if self.locals.get("done") and "episode" in self.locals.get("infos"):
                self.episode_vec_returns.append(self.locals.get("infos")["episode"]["r"])
        return True

    def _on_training_end(self) -> None:
        file_name = f"MORL_{self.env_name}_{self.algo_name}_policy{self.policy_id}_seed{self.seed}_{self.iter}.npz"
        file_path = os.path.join(self.save_dir, file_name)

        np.savez_compressed(file_path, episode_vec_returns=self.episode_vec_returns)

