import os
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
import numpy as np
import traceback

import sys
# Get absolute path to project root
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.insert(0, PROJECT_ROOT)

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), 
    '..',
    '..'
))
sys.path.append(PROJECT_ROOT)

from numerical.solvers.dg_wave_solver import DGWaveSolver
from numerical.environments.dg_amr_env import DGAMREnv

class DebugCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_count = 0
        self.episode_reward = 0
    
    def _on_step(self) -> bool:
        # Accumulate reward
        reward = self.locals.get("rewards")
        if reward is not None:
            self.episode_reward += reward[0]

        # Check if episode ended
        done = self.locals.get("dones")
        if done is not None and done[0]:
            self.episode_count += 1
            print(f"Episode {self.episode_count} completed")
            print(f"Episode reward: {self.episode_reward}")
            print(f"Number of timesteps: {self.num_timesteps}")
            print("-------------------------")
            self.episode_reward = 0  # Reset for next episode
            
        return True

# Initialize solver
nop = 3  # Polynomial order
xelem = np.array([-1.0, -0.4, 0.0, 0.4, 1.0])  # Domain boundaries
max_elements = 25  # Maximum number of elements
max_level = 3  # Maximum refinement level
solver = DGWaveSolver(nop, xelem, max_elements, max_level)

# Create and initialize the environment
env = DGAMREnv(solver, element_budget=25, gamma_c = 25.0)

# Set up logging directory
log_dir = "./tensorboard_logs/"
os.makedirs(log_dir, exist_ok=True)

# Configure logger
new_logger = configure(log_dir, ["tensorboard", "stdout"])

# Initialize the model
model = A2C("MultiInputPolicy", env, verbose=1, tensorboard_log=log_dir)
model.set_logger(new_logger)

# Create callback
callback = DebugCallback()

# Train for just a few episodes
try:
    model.learn(total_timesteps=10, callback=callback)
    print("\nTraining completed successfully")
    print(f"Tensorboard logs written to: {log_dir}")
except Exception as e:
    print(f"Error during training: {e}")
    traceback.print_exc()