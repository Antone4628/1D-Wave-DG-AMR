
#The following are some proposed changes in order to use max_elements AND max_level.

# This approach:

# Uses max_level to calculate dt via dx_min
# Uses max_elements as resource budget for RL agent
# Prevents refinement beyond max_level even if element budget allows
# Maintains stability through proper dt calculation

# class DGWaveSolver:
#     def __init__(self, nop, xelem, max_elements, max_level, icase):
#         self.nop = nop
#         self.xelem = xelem
#         self.max_elements = max_elements
#         self.max_level = max_level  # Keep for dx calculation
#         self.icase = icase
        
#         # Calculate minimum dx based on max_level
#         self.dx_min = np.min(np.diff(xelem)) / (2**max_level)
        
#     def adapt_mesh(self, marks_override=None):
#         # Get current refinement levels for each element
#         current_levels = self.info_mat[self.active - 1, 2]
        
#         # Modify marks to prevent exceeding max_level
#         if marks_override is not None:
#             for elem_idx, mark in marks_override.items():
#                 if mark == 1 and current_levels[elem_idx] >= self.max_level:
#                     marks_override[elem_idx] = 0
        
#         # Check if adaptation would exceed element budget
#         proposed_elem_count = self._compute_proposed_elements(marks_override)
#         if proposed_elem_count > self.max_elements:
#             return False  # Adaptation rejected
            
#         # Proceed with adaptation
#         return self._perform_adaptation(marks_override)



def training_step(self, action, observation):
    """
    Execute one step of training with 2:1 balance enforcement.
    
    Args:
        action: Element adaptation (0=coarsen, 1=no change, 2=refine)
        observation: Current observation
    """
    # Get current element and create initial marks array
    current_elem = self.current_element
    marks = np.zeros(len(self.solver.active))
    marks[current_elem] = action - 1  # Convert to {-1,0,1}
    
    # Enforce 2:1 balance - this may add additional refinement/coarsening marks
    marks = enforce_2_1_balance(
        self.solver.label_mat,
        self.solver.info_mat, 
        self.solver.active,
        marks
    )
    
    # Store initial state for reward calculation
    old_solution = self.solver.q.copy()
    old_grid = self.solver.coord.copy()
    old_resources = len(self.solver.active) / self.solver.max_elements
    
    # Apply all required mesh changes to maintain 2:1 balance
    self.solver.adapt_mesh(marks_override=marks)
    
    # Take solver timestep
    self.solver.step()
    
    # Get new solution
    new_solution = self.solver.q
    new_grid = self.solver.coord
    new_resources = len(self.solver.active) / self.solver.max_elements
    
    # Compute change in solution on finer grid
    if len(new_solution) >= len(old_solution):
        old_interpolated = np.interp(new_grid, old_grid, old_solution)
        delta_u = np.linalg.norm(new_solution - old_interpolated)
    else:
        new_interpolated = np.interp(old_grid, new_grid, new_solution)
        delta_u = np.linalg.norm(new_interpolated - old_solution)
    
    # Compute reward - now accounts for all changes required for 2:1 balance
    reward = self._compute_reward(delta_u, new_resources)
    
    # Get new observation
    observation = self._get_observation()
    
    # Check termination
    terminated = False
    truncated = (new_resources >= 1.0)
    
    # Additional info for debugging
    info = {
        'delta_u': delta_u,
        'resource_usage': new_resources,
        'n_elements': len(self.solver.active),
        'total_changes': np.sum(marks != 0)  # Number of elements changed
    }
    
    # Move to next element randomly
    self.current_element = np.random.randint(0, len(self.solver.active))
    
    return observation, reward, terminated, truncated, info