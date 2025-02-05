"""
dg_amr_env_documented.py

A detailed implementation of a Gymnasium environment for Adaptive Mesh Refinement (AMR) 
using Deep Reinforcement Learning, based on:
'Deep reinforcement learning for adaptive mesh refinement' (Foucart et al., 2023)

This environment allows an RL agent to learn AMR strategies by:
1. Observing local solution properties (jumps, values)
2. Making refinement decisions (coarsen, maintain, or refine elements)
3. Receiving rewards based on solution accuracy vs computational cost

The environment follows the formulation from the paper with observation space
based on solution jumps and resource usage, discrete actions for mesh refinement,
and a reward function balancing accuracy against computational cost.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Tuple, Any
from ..solvers.dg_wave_solver import DGWaveSolver

class DGAMREnv(gym.Env):
    """
    Gymnasium environment for DG Wave AMR using Reinforcement Learning.
    
    The environment wraps a DG Wave solver and provides:
    1. Observation Space:
       - local_jumps: Solution discontinuities at element boundaries
       - neighbor_jumps: Jumps at neighboring elements
       - avg_jump: Average jump across all elements
       - resource_usage: Fraction of computational budget used
       - solution_values: Local solution within current element
       
    2. Action Space:
       - 0: Coarsen element (combine with sibling)
       - 1: Maintain current resolution
       - 2: Refine element (split into children)
       
    3. Reward Function:
       R = accuracy_term - resource_penalty
       where:
       - accuracy_term = log(|Δu| + ε) - log(ε)
       - resource_penalty = γc * sqrt(p)/(1-p)
       - p is fraction of resources used
       - γc controls accuracy vs. cost trade-off
    """
    
    def __init__(
        self,
        solver: DGWaveSolver,
        gamma_c: float = 25.0,
        render_mode: str = None
    ):
        """
        Initialize the AMR environment.
        
        Args:
            solver: DG wave solver instance - provides numerical solution
            gamma_c: Cost penalty coefficient - controls accuracy vs. resource trade-off
                    Higher values prioritize resource efficiency
            render_mode: Visualization mode (if needed for debugging)
        """
        # Initialize parent Gymnasium environment
        super().__init__()
        
        # Store solver and parameters
        self.solver = solver          # DG solver for wave equation
        self.gamma_c = gamma_c        # Cost penalty coefficient
        self.render_mode = render_mode
        self.current_element = 0      # Index of element being considered
        self.machine_eps = 1e-16      # Used for log scaling in reward
        
        # Define action space:
        # Discrete space with 3 actions mapping to {-1, 0, 1} for mesh adaptation
        self.action_space = spaces.Discrete(3)
        
        # Define observation space components:
        self.observation_space = spaces.Dict({
            # Solution jumps at element boundaries and interior nodes
            # Shape is (ngl,) where ngl is number of LGL points per element
            'local_jumps': spaces.Box(
                low=0.0,              # Jumps are absolute values
                high=1e3,             # Upper bound based on solution range
                shape=(self.solver.ngl,), 
                dtype=np.float32
            ),
            
            # Jumps at neighboring elements [left_neighbor, right_neighbor]
            'neighbor_jumps': spaces.Box(
                low=0.0,
                high=1e3,
                shape=(2,),           # One value for each neighbor
                dtype=np.float32
            ),
            
            # Average jump across all mesh elements
            'avg_jump': spaces.Box(
                low=0.0,
                high=1e3,
                shape=(1,),           # Single scalar value
                dtype=np.float32
            ),
            
            # Fraction of computational resources currently used
            'resource_usage': spaces.Box(
                low=0.0,
                high=1.0,             # Represents 0-100% of budget
                shape=(1,),
                dtype=np.float32
            ),
            
            # Local solution values within current element
            'solution_values': spaces.Box(
                low=-1e3,             # Solution can be negative
                high=1e3,
                shape=(self.solver.ngl,),
                dtype=np.float32
            )
        })

def _get_element_jumps(self, element_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute solution discontinuities for current element and its neighbors.
        
        Calculates:
        1. Local jumps at element boundaries and between interior nodes
        2. Interface jumps with neighboring elements
        
        The jumps provide a measure of solution smoothness and are key indicators
        for mesh adaptation decisions.
        
        Args:
            element_idx: Index of element in active_grid
            
        Returns:
            tuple: (local_jumps, neighbor_jumps)
                local_jumps[ngl]: Jumps at each node within element
                neighbor_jumps[2]: Jumps at interfaces with neighbors
        """
        # Get element number from active elements list
        elem = self.solver.active[element_idx]
        
        # Extract solution values for current element
        elem_nodes = self.solver.intma[:, element_idx]  # Global node indices
        elem_sol = self.solver.q[elem_nodes]            # Solution values
        
        # Get boundary values for interface calculations
        elem_left = elem_sol[0]    # Left endpoint solution
        elem_right = elem_sol[-1]  # Right endpoint solution
        
        # Initialize arrays for jump values
        local_jumps = np.zeros(self.solver.ngl)  # One jump per node
        neighbor_jumps = np.zeros(2)             # Left and right interface jumps
        
        # Calculate jumps with left neighbor if it exists
        if elem > 1:  # Not leftmost element
            # Find left neighbor in active elements
            left_active_idx = np.where(self.solver.active == elem-1)[0]
            if len(left_active_idx) > 0:  # Neighbor exists
                left_idx = left_active_idx[0]
                # Get neighbor's solution values
                left_nodes = self.solver.intma[:, left_idx]
                left_sol = self.solver.q[left_nodes]
                # Compute jump at left interface
                local_jumps[0] = abs(elem_left - left_sol[-1])
                neighbor_jumps[0] = local_jumps[0]
                
        # Calculate jumps with right neighbor if it exists
        if elem < len(self.solver.label_mat):  # Not rightmost element
            right_active_idx = np.where(self.solver.active == elem+1)[0]
            if len(right_active_idx) > 0:  # Neighbor exists
                right_idx = right_active_idx[0]
                # Get neighbor's solution values
                right_nodes = self.solver.intma[:, right_idx]
                right_sol = self.solver.q[right_nodes]
                # Compute jump at right interface
                local_jumps[-1] = abs(elem_right - right_sol[0])
                neighbor_jumps[1] = local_jumps[-1]
                
        # Calculate jumps between interior nodes
        for i in range(1, self.solver.ngl-1):
            local_jumps[i] = abs(elem_sol[i] - elem_sol[i-1])
            
        return local_jumps, neighbor_jumps
    
def _get_observation(self) -> Dict[str, np.ndarray]:
    """
    Construct observation vector for current environment state.
    
    Gathers local and global information needed by the RL agent:
    - Solution discontinuities in current element
    - Jumps at neighboring interfaces
    - Average jump across mesh (global smoothness indicator)
    - Resource usage (guides computational efficiency)
    - Local solution values
    
    Returns:
        dict: Components of observation space
    """
    # Get solution jumps for current element
    local_jumps, neighbor_jumps = self._get_element_jumps(self.current_element)
    
    # Compute mean jump across all elements (global smoothness measure)
    all_jumps = []
    for i in range(len(self.solver.active)):
        jumps, _ = self._get_element_jumps(i)
        all_jumps.append(jumps.mean())
    avg_jump = np.mean(all_jumps)
    
    # Calculate fraction of computational budget used
    resource_usage = len(self.solver.active) / self.solver.max_elements
    
    # Get solution values in current element
    element_nodes = self.solver.intma[:, self.current_element]
    solution_values = self.solver.q[element_nodes]
    
    # Construct observation dictionary
    return {
        'local_jumps': local_jumps.astype(np.float32),
        'neighbor_jumps': neighbor_jumps.astype(np.float32),
        'avg_jump': np.array([avg_jump], dtype=np.float32),
        'resource_usage': np.array([resource_usage], dtype=np.float32),
        'solution_values': solution_values.astype(np.float32)
    }
        
def _compute_reward(self, delta_u: float, new_resources: float) -> float:
    """
    Calculate reward balancing accuracy improvement against resource usage.
    
    Follows paper's formulation:
    R = accuracy_term - resource_penalty
    
    accuracy_term = log(|Δu| + ε) - log(ε)
    - Measures solution improvement
    - Log scaling prevents domination by early refinements
    - ε (machine epsilon) handles zero-change case
    
    resource_penalty = γc * sqrt(p)/(1-p)
    - Penalizes computational cost
    - Barrier function approaches infinity as p→1
    - γc controls accuracy vs. cost trade-off
    
    Args:
        delta_u: Change in solution after adaptation
        new_resources: Updated fraction of resources used
        
    Returns:
        float: Combined reward value
    """
    # Calculate accuracy term using log scaling
    accuracy = np.log(abs(delta_u) + self.machine_eps) - np.log(self.machine_eps)
    
    # Calculate resource penalty using barrier function
    resource_penalty = self.gamma_c * np.sqrt(new_resources) / (1 - new_resources)
    
    return float(accuracy - resource_penalty)
        
def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
    """
    Execute one environment step: adapt mesh, evolve solution, compute reward.
    
    Process:
    1. Store current state for reward calculation
    2. Apply AMR action to current element
    3. Advance solution one timestep
    4. Compute reward based on solution change and resource usage
    5. Construct new observation
    6. Select next element for consideration
    
    Args:
        action: Mesh adaptation choice (0=coarsen, 1=no change, 2=refine)
        
    Returns:
        tuple: (observation, reward, terminated, truncated, info)
            observation: New environment state
            reward: Reward for action taken
            terminated: Whether episode naturally ended (always False)
            truncated: Whether resources exceeded (p ≥ 1)
            info: Additional diagnostics
    """
    # Store current state
    old_solution = self.solver.q.copy()
    old_grid = self.solver.coord.copy()
    old_resources = len(self.solver.active) / self.solver.max_elements
    
    # Convert action to AMR marking (-1=coarsen, 0=maintain, 1=refine)
    marks_override = {self.current_element: action - 1}
    self.solver.adapt_mesh(marks_override=marks_override)
    
    # Advance solution
    self.solver.step()
    
    # Get updated state
    new_solution = self.solver.q
    new_grid = self.solver.coord
    
    # Compare solutions (handle different grid sizes)
    if len(new_solution) >= len(old_solution):
        # Project old solution to refined grid
        old_interpolated = np.interp(new_grid, old_grid, old_solution)
        delta_u = np.linalg.norm(new_solution - old_interpolated)
    else:
        # Project new solution to old grid
        new_interpolated = np.interp(old_grid, new_grid, new_solution)
        delta_u = np.linalg.norm(new_interpolated - old_solution)
    
    # Calculate new resource usage
    new_resources = len(self.solver.active) / self.solver.max_elements
    
    # Compute reward
    reward = self._compute_reward(delta_u, new_resources)
    
    # Get observation of new state
    observation = self._get_observation()
    
    # Check termination conditions
    terminated = False  # Episode doesn't end naturally
    truncated = (new_resources >= 1.0)  # End if resources exceeded
    
    # Collect diagnostic information
    info = {
        'delta_u': delta_u,
        'resource_usage': new_resources,
        'n_elements': len(self.solver.active)
    }
    
    # Randomly select next element
    self.current_element = np.random.randint(0, len(self.solver.active))
    
    return observation, reward, terminated, truncated, info
    
def reset(self, seed=None, options=None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Reset environment to initial state for new episode.
    
    Process:
    1. Reset solver to initial mesh and solution
    2. Select random element to start with
    3. Get initial observation
    
    Args:
        seed: Random seed for reproducibility
        options: Additional reset options (unused)
        
    Returns:
        tuple: (observation, info)
            observation: Initial state
            info: Additional information (empty dict)
    """
    # Initialize random generator
    super().reset(seed=seed)
    
    # Reset solver to initial state
    self.solver.reset()
    
    # Select random starting element
    self.current_element = np.random.randint(0, len(self.solver.active))
    
    # Return initial observation
    return self._get_observation(), {}