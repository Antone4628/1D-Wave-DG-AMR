"""
Script to check DG AMR Environment using Gymnasium's environment checker.
"""

import os
import sys
import traceback

# Get absolute path to project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from gymnasium.utils.env_checker import check_env

# Import your modules
from numerical.solvers.dg_wave_solver import DGWaveSolver
from numerical.environments.dg_amr_env import DGAMREnv

def main():
    # Initialize solver
    xelem = np.array([-1, -0.4, 0, 0.4, 1])
    solver = DGWaveSolver(
        nop=4,
        xelem=xelem,
        max_elements=40,
        max_level = 4,
        courant_max=0.1,
        icase=1
    )
  
    
    # Create environment
    env = DGAMREnv(solver=solver, element_budget=25, gamma_c=25.0)
    
    print("Checking environment...")
    try:
        # Run Gymnasium's environment checker
        check_env(env)
        print("Environment check passed!")
        
        # Print environment spaces
        print("\nAction Space:", env.action_space)
        print("\nObservation Space:")
        for key, space in env.observation_space.spaces.items():
            print(f"{key}: {space}")
            
    except Exception as e:
        print("Environment check failed!")
        print("Error:", str(e))
        print("\nFull traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
