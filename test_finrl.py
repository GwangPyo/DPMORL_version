import fire
import pandas as pd
from sbx import PPO, SAC
import os
from stable_baselines3.common.utils import set_random_seed
import mo_gymnasium
import torch
import numpy as np

from MORL_stablebaselines3.envs.wrappers.utility_env_wrapper import MultiEnv_UtilityFunction, ObsInfoWrapper
from MORL_stablebaselines3.utility_function.utility_function_parameterized import Utility_Function_Parameterized
from MORL_stablebaselines3.utility_function.utility_function_programmed import Utility_Function_Programmed
from MORL_stablebaselines3.utility_function.utility_function_programmed import Utility_Function_Linear
from finrl_env import load_env
from logger import ReturnLogger
from utils import DummyVecEnv
import glob
import pickle
import time
from typing import Literal
import os


def reward_dim(env) -> int:
    try:
        return env.reward_dim
    except AttributeError:
        return reward_dim(env.unwrapped)


class UtilityFunctionLoader(object):
    def __init__(self,
                 test_env,
                 reward_two_dim,
                 reward_dim_indices,
                 exp_name,
                 lamda: float = 1e-2,
                 keep_scale: bool = True,
                 linear_utility: bool = False,

                 ):
        utility_dir = 'experiments/' + exp_name
        os.makedirs(utility_dir, exist_ok=True)
        reward_shape = reward_dim(test_env)
        if reward_two_dim:
            reward_shape = 2
        if reward_dim_indices == '':
            reward_dim_indices = list(range(reward_shape))
        else:
            reward_dim_indices = eval(reward_dim_indices)
            reward_shape = len(reward_dim_indices)
        self.reward_dim_indices = reward_dim_indices
        print(f'{reward_dim_indices = }, {reward_shape = }')
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device {DEVICE}")
        if linear_utility:
            utility_class_programmed = Utility_Function_Linear
        else:
            utility_class_programmed = Utility_Function_Programmed

        norm = True

        utility_function = utility_class_programmed(reward_shape=reward_shape, norm=norm, lamda=lamda,
                                                    function_choice=0, keep_scale=keep_scale)
        self.utility_class_programmed = utility_class_programmed
        self.num_utility_programmed = len(utility_function.utility_functions)

        # Load pretrained utility functions
        assert os.path.isdir(
            f'utility-model-selected/dim-{reward_shape}'), 'There is no pretrained utility functions provided. '
        self.num_pretrained_utility = len(glob.glob(f'utility-model-selected/dim-{reward_shape}/*'))
        pretrained_utility_paths = [f'utility-model-selected/dim-{reward_shape}/utility-{i}.pt'
                                    for i in range(self.num_pretrained_utility)]

        pretrained_utility_functions = []
        for path in pretrained_utility_paths:
            model = Utility_Function_Parameterized(reward_shape=reward_shape, norm=norm, lamda=lamda,
                                                   max_weight=0.5, keep_scale=keep_scale, size_factor=1)
            model.load_state_dict(torch.load(path))
            model.eval()
            model = model.cuda()
            pretrained_utility_functions.append(model)
        self.num_utility_pretrained = len(pretrained_utility_functions)
        self.pretrained_utility_functions = pretrained_utility_functions
        self.reward_shape = reward_shape
        self.norm = norm
        self.lamda = lamda
        self.keep_scale = keep_scale

    def get_utility(self, policy_idx):
        if policy_idx < self.num_utility_programmed:
            utility_function = self.utility_class_programmed(reward_shape=self.reward_shape,
                                                             norm=self.norm, lamda=self.lamda,
                                                             function_choice=policy_idx,
                                                             keep_scale=self.keep_scale)
        else:
            utility_function = self.pretrained_utility_functions[policy_idx - self.num_utility_programmed]
        return utility_function


def evaluate(policy_set, test_env, num_eval, reward_dim) -> pd.DataFrame:
    data = []
    for policy_idx, policy in enumerate(policy_set):
        for exec_idx in range(num_eval):
            data_point = { }
            score_vec = run_one_episode(policy, test_env, reward_dim)
            data_point['policy_id'] = policy_idx
            data_point['execution_id'] = exec_idx
            for dim_idx, score in enumerate(score_vec):
                data_point[f'score_{dim_idx}'] = score
            data.append(data_point)
    return pd.DataFrame(data)


def run_one_episode(policy, test_env, reward_dim):
    obs, _ = test_env.reset()
    done = False
    score_vec = np.zeros(shape=reward_dim)
    while not done:
        action, _ = policy.predict(obs)
        obs, reward, done, timeout, info = test_env.step(action)
        score_vec += reward
        done = done or timeout
    return score_vec


class Main(object):
    def __init__(self,
                 reward_two_dim: bool = False,
                 reward_dim_indices: str = '',
                 exp_name: str = 'dpmorl',
                 max_num_policies: int = 20,
                 total_steps: int = int(1e7),
                 iters: int = 50,
                 num_cpu: int = 20,
                 num_eval: int = 100,
                 base_algo: Literal['PPO', 'SAC'] = 'PPO',
                 lamda: float = 1e-2,
                 keep_scale: bool = True,
                 linear_utility: bool = False,
                 max_episode_steps: int = 500,
                 augment_state: bool = False,
                 gpu_id: str = '0',
                 seed: int = 42
                 ):

        env_id = 'finrl'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        utility_dir = 'experiments/' + exp_name
        with open('normalization_data/data.pickle', 'rb') as file:
            self.normalization_data = pickle.load(file)
        self.augment_state = augment_state
        self.num_cpu = num_cpu
        self.env_id = env_id
        self.max_episode_steps = max_episode_steps
        self.test_env = load_env(mode='test',seed=seed)

        self.utility_loader = UtilityFunctionLoader(
            test_env=self.test_env, reward_two_dim=reward_two_dim,
            reward_dim_indices=reward_dim_indices, exp_name=exp_name,
            lamda=lamda, keep_scale=keep_scale, linear_utility=linear_utility
        )
        self.reward_dim_indices = self.utility_loader.reward_dim_indices
        self.policies = []
        self.utility_functions_optims = []

        self.total_steps = total_steps
        self.iters = iters
        pre_zts = None
        task_name = f"DPMORL.{self.env_id}.{'no_norm.' if True == False else ''}LossNormLamda_{lamda}"
        self.utility_dir = os.path.join(utility_dir, task_name)

        self.num_total_policies = min(
            self.utility_loader.num_utility_programmed + self.utility_loader.num_utility_pretrained,
            max_num_policies)
        self.seed = seed
        print(f'{self.num_total_policies = }')
        self.base_algo = base_algo
        self.algo_classes = { "PPO": PPO, "SAC": SAC }
        self.ALGO = self.algo_classes[base_algo]
        algo_kwargs = { "PPO": { "n_epochs": 5 }, "SAC": { } }
        self.algo_kwarg = algo_kwargs[base_algo]
        self.num_eval = num_eval

    @property
    def reward_shape(self):
        return self.utility_loader.reward_shape

    @property
    def env_key(self):
        return ENV_ID_TO_NAME[self.env_id]

    @property
    def reward_dim(self):
        return reward_dim(self.test_env)

    @property
    def num_utility_programmed(self):
        return self.utility_loader.num_utility_programmed

    def run(self):
        learned_policies = []
        for policy_idx in range(self.num_total_policies):
            utility_function = self.utility_loader.get_utility(policy_idx)
            print('normalization data: None')
            optim, optim_init_state = None, None
            self.utility_functions_optims.append([utility_function, optim, optim_init_state])
            env = DummyVecEnv(
                [self.__train_env_builder() for i in range(self.num_cpu)],
                reward_dim=self.reward_shape
            )
            env = MultiEnv_UtilityFunction(env, utility_function, reward_dim=self.reward_shape,
                                           augment_state=self.augment_state)
            env.update_utility_function(utility_function)

            policy = self.ALGO("MlpPolicy", env, verbose=1, device='cuda',
                               **self.algo_kwarg)

            if policy_idx < self.num_utility_programmed:
                policy_name = f'program-{policy_idx}-{self.seed}'
            else:
                policy_name = f'pretrain-{policy_idx - self.num_utility_programmed}-{self.seed}'
            print(f"Training policy {policy_idx + 1} with {self.total_steps} steps...")
            curtime = time.time()
            return_logger = ReturnLogger(self.utility_dir, self.env_id, self.base_algo, policy_name, 0, self.seed)
            policy.learn(total_timesteps=self.total_steps, callback=return_logger, progress_bar=True)
            print(f"Training one policy with one utility function using time {time.time() - curtime:.2f} seconds.")
            policy.save(f'{self.utility_dir}/policy-{policy_name}')
            learned_policies.append(policy)
        test_data = evaluate(learned_policies, self.test_env, self.num_eval, self.reward_dim)
        train_data = evaluate(learned_policies, load_env(mode='train', seed=self.seed,), self.num_eval, self.reward_dim)
        os.makedirs('results', exist_ok=True)
        os.makedirs(f'results/{self.env_id}', exist_ok=True)
        test_data.to_csv(f'results/{self.env_id}/DPMORL_{self.seed}.csv')
        train_data.to_csv(f'results/{self.env_id}/DPMORL_{self.seed}.csv')

    def __train_env_builder(self, ):

        def _init():
            env = load_env(mode='train', seed=self.seed)
            env.name = self.env_id
            env = ObsInfoWrapper(env, reward_dim=self.reward_dim, reward_dim_indices=self.reward_dim_indices)
            return env

        set_random_seed(self.seed)
        return _init


def cli(**kwargs):
    return Main(**kwargs).run()


if __name__ == '__main__':
    fire.Fire(cli)
