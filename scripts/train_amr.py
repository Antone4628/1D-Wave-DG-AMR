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

# Import your modules - adjust these paths based on your project structure
# Assuming scripts/ is at the same level as numerical/
import sys
sys.path.append('..')  # Add parent directory to path

from numerical.solvers.dg_wave_solver import DGWaveSolver
from numerical.environments.dg_amr_env import DGAMREnv

def make_env():
    """Create and initialize environment."""
    # Initialize solver
    xelem = np.array([-1, -0.4, 0, 0.4, 1])
    solver = DGWaveSolver(
        nop=4,
        xelem=xelem,
        max_level=4,
        icase=1
    )
    
    # Create environment
    env = DGAMREnv(solver=solver, gamma_c=25.0)
    
    # Wrap environment for training
    env = Monitor(env)
    
    return env

def train_a2c():
    """Train A2C model for AMR."""
    # Create environment
    env = make_env()
    
    # Verify environment
    check_env(env)
    
    # Wrap in vectorized environment (required by Stable-Baselines3)
    env = DummyVecEnv([lambda: env])
    
    # Normalize observations and rewards
    env = VecNormalize(
        env,
        norm_obs=True,      # Normalize observations
        norm_reward=True,   # Normalize rewards
        clip_obs=10.,       # Clip observations
        clip_reward=10.,    # Clip rewards
    )
    
    # Create model with custom network architecture
    policy_kwargs = dict(
        # Following paper's architecture
        net_arch=dict(
            pi=[64, 64],  # Actor network
            vf=[64, 64]   # Critic network
        )
    )
    
    model = A2C(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=1e-4,           # Learning rate from paper
        n_steps=5,                    # Number of steps between updates
        gamma=0.99,                   # Discount factor
        gae_lambda=0.95,              # GAE parameter
        ent_coef=0.01,                # Entropy coefficient
        vf_coef=0.5,                  # Value function coefficient
        max_grad_norm=0.5,            # Max gradient norm
        use_rms_prop=True,            # Use RMSProp optimizer
        rms_prop_eps=1e-5,            # RMSProp epsilon
        verbose=1
    )
    
    # Set up callbacks
    # Save best model based on mean reward
    eval_callback = EvalCallback(
        env,
        best_model_save_path='./logs/best_model',
        log_path='./logs/eval_results',
        eval_freq=1000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )
    
    # Periodically save model during training
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path='./logs/checkpoints/',
        name_prefix='amr_model'
    )
    
    # Train model
    total_timesteps = 200000  # Adjust based on your needs
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback]
    )
    
    # Save final model
    model.save("amr_a2c_final")
    
    return model, env

def evaluate_model(model, env, n_episodes=10):
    """Evaluate trained model."""
    rewards = []
    
    for episode in range(n_episodes):
        obs = env.reset()[0]
        episode_reward = 0
        done = False
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            
            # Track metrics
            print(f"Elements: {info[0]['n_elements']}, Resource usage: {info[0]['resource_usage']:.3f}")
            
        rewards.append(episode_reward)
        print(f"Episode {episode+1} reward: {episode_reward}")
    
    print(f"\nMean reward over {n_episodes} episodes: {np.mean(rewards):.3f}")
    print(f"Std dev of rewards: {np.std(rewards):.3f}")

if __name__ == "__main__":
    # Create logging directory
    import os
    os.makedirs('./logs', exist_ok=True)
    
    # Train model
    print("Starting training...")
    model, env = train_a2c()
    
    # Evaluate model
    print("\nEvaluating model...")
    evaluate_model(model, env)