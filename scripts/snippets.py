
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


#This code implements the enforce_2_1_balance
def adapt_mesh(self, criterion=1, marks_override=None):
    """
    Unified mesh adaptation routine that handles both refinement and derefinement.
    Enforces 2:1 balance between element refinement levels.
    
    Args:
        criterion: AMR marking criterion selector
        marks_override: Optional predefined marks to override automatic marking
    
    Returns:
        tuple: (adapted grid, active cells, new element count, 
               new CG point count, new DG point count)
    """
    ngl = self.nop + 1
    
    # Get refinement marks
    if marks_override is not None:
        marks = marks_override
    else:
        marks = mark(self.active, self.label_mat, self.intma, self.q, criterion)
    
    # Early exit if no adaptation needed
    if not np.any(marks):
        new_nelem = len(self.active)
        return (self.coord, self.active, marks, new_nelem, 
                self.nop * new_nelem + 1, ngl * new_nelem)
    
    # Enforce 2:1 balance
    marks = enforce_2_1_balance(self.label_mat, self.active, marks)
    
    # Process adaptations one at a time
    i = 0
    while i < len(marks):
        if marks[i] == 0:
            i += 1
            continue
            
        if marks[i] > 0:
            # Handle refinement
            elem = self.active[i]
            parent_idx = elem - 1
            c1, c2 = self.label_mat[parent_idx][2:4]
            c1_r = self.info_mat[c1-1][4]
            
            # Update grid
            self.coord = np.insert(self.coord, i + 1, c1_r)
            
            # Update active cells and marks
            self.active = np.concatenate([
                self.active[:i],
                [c1, c2],
                self.active[i+1:]
            ])
            
            marks = np.concatenate([
                marks[:i],
                [0, 0],
                marks[i+1:]
            ])
            
            # Skip the newly added element
            i += 2
            
        else:  # marks[i] < 0
            # Handle derefinement
            elem = self.active[i]
            parent = self.label_mat[elem-1][1]
            
            # Find sibling
            if self.label_mat[elem-2][1] == parent and i > 0 and marks[i-1] < 0:
                # Sibling is previous element
                sib_idx = i - 1
                min_idx = sib_idx
            elif i + 1 < len(marks) and self.label_mat[elem][1] == parent and marks[i+1] < 0:
                # Sibling is next element
                sib_idx = i + 1
                min_idx = i
            else:
                # No valid sibling found for derefinement
                i += 1
                continue
                
            # Remove grid point between elements
            self.coord = np.delete(self.coord, min_idx + 1)
            
            # Update active cells and marks
            self.active = np.concatenate([
                self.active[:min_idx],
                [parent],
                self.active[min_idx+2:]
            ])
            
            marks = np.concatenate([
                marks[:min_idx],
                [0],
                marks[min_idx+2:]
            ])
            
            # Continue checking from the position after the derefined pair
            i = min_idx + 1
    
    # Calculate new dimensions
    new_nelem = len(self.active)
    new_npoin_cg = self.nop * new_nelem + 1
    new_npoin_dg = ngl * new_nelem
    
    return self.coord, self.active, marks, new_nelem, new_npoin_cg, new_npoin_dg

def enforce_2_1_balance(label_mat, active, marks):
    """
    Enforces 2:1 balance by propagating refinement as needed.
    Uses level information directly from label_mat[:,4].
    """
    from collections import deque
    
    # Keep track of cells we've processed
    processed = set()
    
    # Process queue
    queue = deque()
    for i, mark in enumerate(marks):
        if mark == 1:  # Initially add all refinement marks
            queue.append(i)
            
    while queue:
        idx = queue.popleft()
        if idx in processed:
            continue
            
        elem = active[idx]
        elem_level = label_mat[elem-1][4]  # Get level directly from label_mat
        
        # Get neighbors
        neighbors = get_element_neighbors(elem, label_mat, active)
        
        # Check each neighbor
        for neighbor in neighbors:
            if neighbor is None:
                continue
                
            neighbor_idx = np.where(active == neighbor)[0][0]
            neighbor_level = label_mat[neighbor-1][4]  # Get level from label_mat
            
            # If neighbor would be more than 1 level coarser after refinement
            if elem_level + 1 - neighbor_level > 1:
                # Mark neighbor for refinement
                marks[neighbor_idx] = 1
                queue.append(neighbor_idx)
                
        processed.add(idx)
        
    # Now check if any coarsening would violate 2:1 balance
    for i, mark in enumerate(marks):
        if mark == -1:
            elem = active[i]
            elem_level = label_mat[elem-1][4]
            
            # Check if this element can be coarsened
            parent = label_mat[elem-1][1]
            if parent == 0:  # Can't coarsen root elements
                marks[i] = 0
                continue
                
            # Find sibling
            siblings = []
            if elem > 1 and label_mat[elem-2][1] == parent:
                siblings.append(elem-1)
            if elem < len(label_mat) and label_mat[elem][1] == parent:
                siblings.append(elem+1)
            
            # Both siblings must be marked for coarsening
            sibling_marked = all(
                s in active and marks[np.where(active == s)[0][0]] == -1 
                for s in siblings
            )
            if not sibling_marked:
                marks[i] = 0
                continue
                
            # Check neighbors of both this element and siblings
            all_neighbors = []
            all_neighbors.extend(get_element_neighbors(elem, label_mat, active))
            for sib in siblings:
                if sib in active:
                    all_neighbors.extend(get_element_neighbors(sib, label_mat, active))
            
            # Remove duplicates and None values
            all_neighbors = [n for n in set(all_neighbors) if n is not None]
            
            # Check if coarsening would violate 2:1 balance with any neighbor
            for neighbor in all_neighbors:
                neighbor_level = label_mat[neighbor-1][4]
                
                # If neighbor is too refined relative to coarsened element
                if neighbor_level - (elem_level - 1) > 1:
                    marks[i] = 0  # Prevent coarsening
                    break
                    
    return marks