#!/usr/bin/env python

"""
Training script for DG AMR using A2C.
Implements training procedure following Foucart et al. (2023).
"""

import os
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

import sys
# Get absolute path to project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
from numerical.solvers.dg_wave_solver import DGWaveSolver
from numerical.environments.dg_amr_env import DGAMREnv

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self):
        # Log environment info
        info = self.locals['infos'][0]
        self.logger.record('environment/n_elements', info['n_elements'])
        self.logger.record('environment/resource_usage', info['resource_usage'])
        
        # Log episode rewards
        self.logger.record('rewards/ep_rew_mean', self.locals['ep_info_buffer'][0]['r']) if len(self.locals['ep_info_buffer']) > 0 else None
        
        # Log losses
        self.logger.record('losses/value_loss', self.locals['values_losses'][0])
        self.logger.record('losses/policy_loss', self.locals['policy_loss'][0])
        
        self.logger.dump(step=self.num_timesteps)
        return True

def make_env():
    xelem = np.array([-1, -0.4, 0, 0.4, 1])
    solver = DGWaveSolver(
        nop=4,
        xelem=xelem,
        max_elements = 25,
        max_level=4,
        courant_max=0.1,
        icase=1
    )
    
    env = DGAMREnv(solver=solver, element_budget=25, gamma_c=25.0)
    env = Monitor(env)
    
    return env

def train_a2c(total_timesteps=1000):
    env = make_env()
    check_env(env)
    env = DummyVecEnv([lambda: env])
    
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.,
        clip_reward=10.,
    )
    
    policy_kwargs = dict(
        net_arch=dict(
            pi=[64, 64],
            vf=[64, 64]
        )
    )
    
    model = A2C(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=1e-3,
        n_steps=5,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_rms_prop=True,
        rms_prop_eps=1e-5,
        verbose=1,
        tensorboard_log="./logs/tensorboard/"
    )
    
    eval_callback = EvalCallback(
        env,
        best_model_save_path='./logs/best_model',
        log_path='./logs/eval_results',
        eval_freq=100,
        n_eval_episodes=2,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=500,
        save_path='./logs/checkpoints/',
        name_prefix='amr_model'
    )
    
    tensorboard_callback = TensorboardCallback()
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback, tensorboard_callback]
    )
    
    model.save("amr_a2c_final")
    
    return model, env

def evaluate_model(model, env, n_episodes=10):
    rewards = []
    
    for episode in range(n_episodes):
        obs = env.reset()[0]
        episode_reward = 0
        done = False
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            
            print(f"Elements: {info[0]['n_elements']}, Resource usage: {info[0]['resource_usage']:.3f}")
            
        rewards.append(episode_reward)
        print(f"Episode {episode+1} reward: {episode_reward}")
    
    print(f"\nMean reward over {n_episodes} episodes: {np.mean(rewards):.3f}")
    print(f"Std dev of rewards: {np.std(rewards):.3f}")

if __name__ == "__main__":
    os.makedirs('./logs', exist_ok=True)
    os.makedirs('./logs/tensorboard', exist_ok=True)
    
    print("Starting training...")
    model, env = train_a2c()
    
    print("\nEvaluating model...")
    evaluate_model(model, env)