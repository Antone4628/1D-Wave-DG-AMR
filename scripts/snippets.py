
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