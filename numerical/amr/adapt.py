import numpy as np
from .forest import get_active_levels
from ..grid.mesh import create_grid_us

#~~~~~~~~~~~~~~~~~~~ single refine/coarsen routine ~~~~~~~~~~~~~~~~~~~~~~~~

def mark(active_grid, label_mat, intma, q, criterion):
    """
    Mark elements for refinement/coarsening based on solution criteria.
    
    Args:
        active_grid (array): Currently active element indices
        label_mat (array): Element family relationships [rows, 4]
        intma (array): Element-node connectivity
        q (array): Solution values
        criterion (int): Determines which marking criterion to use
            - criterion = 1: refine when solution > 0.5. Used for building out AMR
            - criterion = 2: Brennan's Criterion
        
    Returns:
        marks (array): Element markers (-1:derefine, 0:no change, 1:refine)
            
    Notes:
        - Elements with solution values >= 0.5 are marked for refinement
        - Elements with solution values < 0.5 are marked for derefinement
        - Derefinement only occurs if both siblings meet criteria
    """
    n_active = len(active_grid)
    marks = np.zeros(n_active, dtype=int)
    refs = []
    defs = []
    
    # Pre-compute label matrix lookups
    parents = label_mat[active_grid - 1, 1]
    children = label_mat[active_grid - 1, 2:4]
    
    # Process each active element
    for idx, (elem, parent) in enumerate(zip(active_grid, parents)):
        # Get element solution values
        elem_nodes = intma[:, idx]
        elem_sols = q[elem_nodes]
        max_sol = np.max(elem_sols)
        
        # Check refinement criteria
        if (criterion == 1):
            if max_sol >= 0.5 and children[idx, 0] != 0:
            # if max_sol >= - 0.5 and children[idx, 0] != 0:
                refs.append(elem)
                marks[idx] = 1
                continue
                
            # Check coarsening criteria
            if max_sol < 0.5 and parent != 0:
                # Find sibling
                sibling = None
                if elem > 1 and label_mat[elem-2, 1] == parent:
                    sibling = elem - 1
                    sib_idx = idx - 1
                elif elem < len(label_mat) and label_mat[elem, 1] == parent:
                    sibling = elem + 1
                    sib_idx = idx + 1
                    
                # Verify sibling status
                if sibling in active_grid:
                    sib_nodes = intma[:, sib_idx]
                    sib_sols = q[sib_nodes]
                    
                    # Mark for coarsening if sibling also qualifies
                    if np.max(sib_sols) < 0.5 and sibling not in defs:
                        marks[idx] = marks[sib_idx] = -1
                        defs.extend([elem, sibling])
        
        if (criterion == 2):
            #Brennan will create criterion here
            pass

    
    return  marks



def check_balance(active_grid, label_mat):
    """
    Check balance of active grid. 
    Returns True if balanced, False otherwise.
    """
    levels = get_active_levels(active_grid, label_mat)
    balance_status = np.all(np.abs(np.diff(levels)) <= 1)
    return balance_status

def balance_mark(active, label_mat):
    n_active = len(active)
    balance_marks = np.zeros(n_active, dtype=int)
    levels = get_active_levels(active, label_mat)

    for e in range(n_active):
        # Get level of element and left neighbor
        elem_level = levels[e]
        left_level = levels[e-1]
        # Check level difference between elements
        level_difference = abs(elem_level - left_level)

        if level_difference > 1:
            # Mark loest level element for refinement
            if elem_level > left_level:
                #left level is lower, mark left element for refinement
                balance_marks[e-1] = 1
            elif elem_level < left_level:
                #elem level is lower, mark current element for refinement
                balance_marks[e] = 1

    return balance_marks


def enforce_balance(active, label_mat, grid, info_mat, nop, coord, PS1, PS2, PG1, PG2, ngl, xgl, qp, max_level):
    """
    Single step 2:1 Balance enforcement. 
    This routine handles balance marking, mesh adapting, and solution adapting.
    This routine is called only once per adaptation step.
    """
    bal_ctr = 0
    while (bal_ctr <= max_level):
        if check_balance(active, label_mat):
            # print(f'grid is balanced. level: {level}')
            bal_ctr = max_level + 1
        else:
            # print(f'balancing grid. balance step: {bal_ctr}')


            bal_marks = balance_mark(active, label_mat)
            pre_active = active  
            pre_grid = grid
            pre_coord = coord

            bal_grid, bal_active, ref_marks, bal_nelem, npoin_cg, bal_npoin_dg = adapt_mesh(nop, grid, active, label_mat, info_mat, bal_marks)
            bal_coord, bal_intma, bal_periodicity = create_grid_us(ngl, bal_nelem, npoin_cg, bal_npoin_dg, xgl, bal_grid)
            bal_q = adapt_sol(qp, pre_coord, bal_marks, pre_active, label_mat, PS1, PS2, PG1, PG2, ngl)

            # Update for next level
            qp = bal_q
            active = bal_active
            nelem = bal_nelem
            intma = bal_intma
            coord = bal_coord
            grid = bal_grid
            npoin_dg = bal_npoin_dg
            periodicity = bal_periodicity

            bal_ctr += 1

    return bal_q, bal_active, bal_nelem, bal_intma, bal_coord, bal_grid, bal_npoin_dg, bal_periodicity

def adapt_mesh(nop, cur_grid, active, label_mat, info_mat, marks):
    """
    Unified mesh adaptation routine that handles both refinement and derefinement.
    
    Args:
        nop: Number of points
        cur_grid: Current grid coordinates
        active: Active cells array
        label_mat: Matrix containing parent-child relationships
        info_mat: Matrix containing cell information
        marks: Array indicating refinement (-1: derefine, 0: no change, 1: refine)
    
    Returns:
        tuple: (adapted grid, active cells, new element count, 
               new CG point count, new DG point count)
    """
    ngl = nop + 1
    
    # Early exit if no adaptation needed
    if not np.any(marks):
        new_nelem = len(active)
        return (cur_grid, active, marks,  new_nelem, 
                nop * new_nelem + 1, ngl * new_nelem)
    
    # Process adaptations one at a time
    i = 0
    while i < len(marks):
        if marks[i] == 0:
            i += 1
            continue
            
        if marks[i] > 0:
            # Handle refinement
            elem = active[i]
            # print(f'refining element {elem}')
            parent_idx = elem - 1
            c1, c2 = label_mat[parent_idx][2:4]
            # print(f'elemenet {elem} has children {c1} and {c2} ')
            # c1_r = info_mat[c1-1][3]
            c1_r = info_mat[c1-1][4]
            
            # Update grid
            cur_grid = np.insert(cur_grid, i + 1, c1_r)
            
            # Update active cells and marks
            active = np.concatenate([
                active[:i],
                [c1, c2],
                active[i+1:]
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
            elem = active[i]
            parent = label_mat[elem-1][1]
            
            # Find sibling
            if label_mat[elem-2][1] == parent and i > 0 and marks[i-1] < 0:
                # Sibling is previous element
                sib_idx = i - 1
                min_idx = sib_idx
            elif i + 1 < len(marks) and label_mat[elem][1] == parent and marks[i+1] < 0:
                # Sibling is next element
                sib_idx = i + 1
                min_idx = i
            else:
                # No valid sibling found for derefinement
                i += 1
                continue
                
            # Remove grid point between elements
            cur_grid = np.delete(cur_grid, min_idx + 1)
            
            # Update active cells and marks
            active = np.concatenate([
                active[:min_idx],
                [parent],
                active[min_idx+2:]
            ])
            
            marks = np.concatenate([
                marks[:min_idx],
                [0],
                marks[min_idx+2:]
            ])
            
            # Continue checking from the position after the derefined pair
            i = min_idx + 1
    
    # Calculate new dimensions
    new_nelem = len(active)
    new_npoin_cg = nop * new_nelem + 1
    new_npoin_dg = ngl * new_nelem
    
    return cur_grid, active, marks, new_nelem, new_npoin_cg, new_npoin_dg


def adapt_sol(q, coord, marks, active, label_mat, PS1, PS2, PG1, PG2, ngl):
    """
    Adapts solution values during mesh adaptation using scatter/gather operations.
    
    Args:
        q (array): Current solution values
        marks (array): Original refinement markers (-1: coarsen, 0: no change, 1: refine)
        active (array): Original (pre-refinement) active element indices. Must correspond 
                       to the original mesh that marks refers to, not the adapted mesh.
        label_mat (array): Element family relationships [elem, parent, child1, child2]
        PS1, PS2 (array): Scatter matrices for child 1 and 2 [ngl, ngl]
        PG1, PG2 (array): Gather matrices for child 1 and 2 [ngl, ngl]
        ngl (int): Number of LGL points per element
        
    Returns:
        array: Adapted solution values
    """
    
    new_q = []
    
    i = 0
    while i < len(marks):
        # print(f"\nProcessing mark {i} for original active element {active[i]}:")
        # print(f"Mark value: {marks[i]}")
        
        if marks[i] == 0:
            # No adaptation - copy solution values
            elem_vals = q[i*ngl:(i+1)*ngl]
            new_q.extend(elem_vals)
            i += 1
            
        elif marks[i] == 1:
            # Refinement - scatter parent solution to children
            parent_elem = active[i]
            # Get parent solution values
            parent_vals = q[i*ngl:(i+1)*ngl]
            
            # Get children from label_mat using original element number
            child1, child2 = label_mat[parent_elem-1][2:4]
            
            # Scatter to get child solutions using mesh-independent matrices
            child1_vals = PS1 @ parent_vals
            child2_vals = PS2 @ parent_vals
            
            # Add both children's solutions
            new_q.extend(child1_vals)
            new_q.extend(child2_vals)
            i += 1
            
        else:  # marks[i] == -1
            # Handle coarsening
            if i + 1 < len(marks) and marks[i+1] == -1:
                # Get the original elements we're coarsening
                child1_elem = active[i]
                child2_elem = active[i+1]
                
                # Get parent from label_mat using original element number
                parent = label_mat[child1_elem-1][1]  # Both children have same parent

                # Get values for both children
                child1_vals = q[i*ngl:(i+1)*ngl]
                child2_vals = q[(i+1)*ngl:(i+2)*ngl]
                
                # Gather children solutions to parent using mesh-independent matrices
                parent_vals = PG1 @ child1_vals + PG2 @ child2_vals
                
                # Add parent solution
                new_q.extend(parent_vals)
                
                # Skip both coarsening marks
                i += 2
            else:
                # print(f"Warning: Unpaired coarsening mark at original element {active[i]}")
                elem_vals = q[i*ngl:(i+1)*ngl]
                new_q.extend(elem_vals)
                i += 1
    
    result = np.array(new_q)
    # print(f"\nFinal adapted solution shape: {result.shape}")
    return result

# def get_element_neighbors(elem, label_mat, active):
#     """
#     Gets neighboring elements by checking siblings and parent/child relationships.
    
#     Returns list of neighbor element numbers or None for boundaries.
#     """
#     neighbors = []
    
#     # Get parent
#     parent = label_mat[elem-1][1]
    
#     # Check previous element
#     if elem > 1 and (elem-1) in active:
#         prev_parent = label_mat[elem-2][1]
#         if prev_parent == parent:  # Share same parent (siblings)
#             neighbors.append(elem-1)
#         else:
#             neighbors.append(elem-1)
            
#     # Check next element
#     if elem < len(label_mat) and (elem+1) in active:
#         next_parent = label_mat[elem][1]
#         if next_parent == parent:  # Share same parent (siblings)
#             neighbors.append(elem+1)
#         else:
#             neighbors.append(elem+1)
            
#     return neighbors
def find_active_neighbor(elem, direction, label_mat, active):
        parent = label_mat[elem-1][1]
        
        if parent == 0:
            if direction == 'left' and elem > 1:
                return elem - 1 if (elem - 1) in active else None
            elif direction == 'right' and elem < len(label_mat):
                return elem + 1 if (elem + 1) in active else None
            return None
            
        parent_children = label_mat[parent-1][2:4]
        if parent_children[0] == 0:  # Invalid children
            return None
            
        # Left child case
        if elem == parent_children[0]:
            if direction == 'left':
                parent_neighbor = find_active_neighbor(parent, 'left', label_mat, active)
                if parent_neighbor is None or parent_neighbor not in active:
                    return None
                neighbor_children = label_mat[parent_neighbor-1][2:4]
                if neighbor_children[0] == 0:
                    return None
                if neighbor_children[1] in active:
                    return neighbor_children[1]
                if neighbor_children[0] in active:
                    return neighbor_children[0]
                return parent_neighbor if parent_neighbor in active else None
            else:  # right
                return parent_children[1] if parent_children[1] in active else None
        
        # Right child case
        else:
            if direction == 'right':
                parent_neighbor = find_active_neighbor(parent, 'right',label_mat, active)
                if parent_neighbor is None or parent_neighbor not in active:
                    return None
                neighbor_children = label_mat[parent_neighbor-1][2:4]
                if neighbor_children[0] == 0:
                    return None
                if neighbor_children[0] in active:
                    return neighbor_children[0]
                if neighbor_children[1] in active:
                    return neighbor_children[1]
                return parent_neighbor if parent_neighbor in active else None
            else:  # left
                return parent_children[0] if parent_children[0] in active else None

def enforce_2_1_balance(label_mat, active, marks):
    """
    Two-stage balance enforcement:
    1. Fix any existing violations
    2. Prevent new violations from marks
    """
    def get_element_index(elem, active_array):
        indices = np.where(active_array == elem)[0]
        return indices[0] if len(indices) > 0 else None
        
    # def find_active_neighbor(elem, direction):
    #     parent = label_mat[elem-1][1]
        
    #     if parent == 0:
    #         if direction == 'left' and elem > 1:
    #             return elem - 1 if (elem - 1) in active else None
    #         elif direction == 'right' and elem < len(label_mat):
    #             return elem + 1 if (elem + 1) in active else None
    #         return None
            
    #     parent_children = label_mat[parent-1][2:4]
    #     if parent_children[0] == 0:  # Invalid children
    #         return None
            
    #     # Left child case
    #     if elem == parent_children[0]:
    #         if direction == 'left':
    #             parent_neighbor = find_active_neighbor(parent, 'left')
    #             if parent_neighbor is None or parent_neighbor not in active:
    #                 return None
    #             neighbor_children = label_mat[parent_neighbor-1][2:4]
    #             if neighbor_children[0] == 0:
    #                 return None
    #             if neighbor_children[1] in active:
    #                 return neighbor_children[1]
    #             if neighbor_children[0] in active:
    #                 return neighbor_children[0]
    #             return parent_neighbor if parent_neighbor in active else None
    #         else:  # right
    #             return parent_children[1] if parent_children[1] in active else None
        
    #     # Right child case
    #     else:
    #         if direction == 'right':
    #             parent_neighbor = find_active_neighbor(parent, 'right')
    #             if parent_neighbor is None or parent_neighbor not in active:
    #                 return None
    #             neighbor_children = label_mat[parent_neighbor-1][2:4]
    #             if neighbor_children[0] == 0:
    #                 return None
    #             if neighbor_children[0] in active:
    #                 return neighbor_children[0]
    #             if neighbor_children[1] in active:
    #                 return neighbor_children[1]
    #             return parent_neighbor if parent_neighbor in active else None
    #         else:  # left
    #             return parent_children[0] if parent_children[0] in active else None
    
    def get_neighbors(elem):
        neighbors = []
        for direction in ['left', 'right']:
            neighbor = find_active_neighbor(elem, direction, label_mat, active)
            if neighbor is not None and neighbor in active:
                neighbors.append(neighbor)
        return neighbors
    
    def fix_existing_violations():
        """Fix any existing 2:1 violations in the mesh."""
        fixed_marks = np.zeros(len(marks), dtype=int)
        
        # Check each active element
        for i, elem in enumerate(active):
            elem_level = label_mat[elem-1][4]
            neighbors = get_neighbors(elem)
            
            for neighbor in neighbors:
                neighbor_idx = get_element_index(neighbor, active)
                if neighbor_idx is None:
                    continue
                    
                neighbor_level = label_mat[neighbor-1][4]
                level_diff = abs(elem_level - neighbor_level)
                
                # If violation exists
                if level_diff > 1:
                    # Refine the coarser element
                    if elem_level > neighbor_level:
                        fixed_marks[neighbor_idx] = 1
                    else:
                        fixed_marks[i] = 1
        
        return fixed_marks
    
    def prevent_new_violations(current_marks):
        """Ensure no new violations would be created."""
        modified_marks = current_marks.copy()
        
        # First handle refinements
        for i, mark in enumerate(modified_marks):
            if mark != 1:
                continue
                
            elem = active[i]
            elem_level = label_mat[elem-1][4]
            neighbors = get_neighbors(elem)
            
            for neighbor in neighbors:
                neighbor_idx = get_element_index(neighbor, active)
                if neighbor_idx is None:
                    continue
                    
                neighbor_level = label_mat[neighbor-1][4]
                
                # If refining would create violation
                if (elem_level + 1) - neighbor_level > 1:
                    # Force neighbor to refine first
                    modified_marks[neighbor_idx] = 1
        
        # Then handle coarsening
        for i, mark in enumerate(modified_marks):
            if mark != -1:
                continue
                
            elem = active[i]
            elem_level = label_mat[elem-1][4]
            parent = label_mat[elem-1][1]
            
            # Can't coarsen root elements
            if parent == 0:
                modified_marks[i] = 0
                continue
                
            # Find sibling
            parent_children = label_mat[parent-1][2:4]
            sibling = parent_children[1] if elem == parent_children[0] else parent_children[0]
            sibling_idx = get_element_index(sibling, active)
            
            # Both siblings must be marked for coarsening
            if sibling_idx is None or modified_marks[sibling_idx] != -1:
                modified_marks[i] = 0
                continue
                
            # Check all neighbors
            neighbors = set()
            for e in [elem, sibling]:
                neighbors.update(get_neighbors(e))
                
            # Remove self and sibling
            neighbors.discard(elem)
            neighbors.discard(sibling)
            
            # Check if coarsening would create violation
            for neighbor in neighbors:
                neighbor_level = label_mat[neighbor-1][4]
                if abs(neighbor_level - (elem_level - 1)) > 1:
                    modified_marks[i] = 0
                    modified_marks[sibling_idx] = 0
                    break
        
        return modified_marks
    
    # Stage 1: Fix any existing violations
    fixed_marks = fix_existing_violations()
    
    # Stage 2: Add user's marks and prevent new violations
    final_marks = np.maximum(fixed_marks, marks)  # Combine fix marks with user marks
    final_marks = prevent_new_violations(final_marks)
    
    return final_marks

def check_2_1_balance(active, label_mat):
    """
    Checks if any elements violate 2:1 balance constraint,
    using tree structure to find true neighbors.
    
    Args:
        active (array): Currently active elements
        label_mat (array): Element family relationships [elem, parent, child1, child2, level]
        
    Returns:
        list: List of tuples (elem1, elem2, level1, level2) for each violation found
    """
    violations = []
    
    # def find_active_neighbor(elem, direction):
    #     """
    #     Find the active neighbor of an element in a given direction.
    #     direction: 'left' or 'right'
    #     """
    #     # Get current element's parent
    #     parent = label_mat[elem-1][1]
        
    #     # If we're at root level, check adjacent root element
    #     if parent == 0:
    #         if direction == 'left' and elem > 1:
    #             return elem - 1 if (elem - 1) in active else None
    #         elif direction == 'right' and elem < len(label_mat):
    #             return elem + 1 if (elem + 1) in active else None
    #         return None
            
    #     # Get parent's children
    #     parent_children = label_mat[parent-1][2:4]
        
    #     # If we're the left child
    #     if elem == parent_children[0]:
    #         if direction == 'left':
    #             # Need to find neighbor to the left of parent
    #             parent_neighbor = find_active_neighbor(parent, 'left')
    #             if parent_neighbor is None:
    #                 return None
    #             # Get rightmost active child of parent's left neighbor
    #             neighbor_children = label_mat[parent_neighbor-1][2:4]
    #             if neighbor_children[1] in active:
    #                 return neighbor_children[1]
    #             if neighbor_children[0] in active:
    #                 return neighbor_children[0]
    #             return parent_neighbor if parent_neighbor in active else None
    #         else:  # direction == 'right'
    #             return parent_children[1] if parent_children[1] in active else parent
                
    #     # If we're the right child
    #     else:
    #         if direction == 'right':
    #             # Need to find neighbor to the right of parent
    #             parent_neighbor = find_active_neighbor(parent, 'right')
    #             if parent_neighbor is None:
    #                 return None
    #             # Get leftmost active child of parent's right neighbor
    #             neighbor_children = label_mat[parent_neighbor-1][2:4]
    #             if neighbor_children[0] in active:
    #                 return neighbor_children[0]
    #             if neighbor_children[1] in active:
    #                 return neighbor_children[1]
    #             return parent_neighbor if parent_neighbor in active else None
    #         else:  # direction == 'left'
    #             return parent_children[0] if parent_children[0] in active else parent
    
    # Check each active element
    for elem in active:
        elem_level = label_mat[elem-1][4]
        
        # Check both left and right neighbors
        for direction in ['left', 'right']:
            neighbor = find_active_neighbor(elem, direction, label_mat, active)
            if neighbor is not None:
                neighbor_level = label_mat[neighbor-1][4]
                level_diff = abs(elem_level - neighbor_level)
                
                if level_diff > 1:
                    violations.append((elem, neighbor, elem_level, neighbor_level))
    
    return violations

def print_balance_violations(active, label_mat):
    """
    Prints any 2:1 balance violations in a readable format.
    """
    violations = check_2_1_balance(active, label_mat)
    if violations:
        print("\n2:1 Balance Violations Found:")
        print("Element  Neighbor  Elem_Level  Neigh_Level")
        print("-----------------------------------------")
        for elem, neighbor, level1, level2 in sorted(violations, key=lambda x: x[0]):
            print(f"{elem:7d}  {neighbor:8d}  {level1:10d}  {level2:11d}")
    else:
        print("\nNo 2:1 balance violations found.")

def print_mesh_state(active, label_mat):
    """
    Prints current mesh state with levels and neighbors.
    """
    print("\nMesh State:")
    print("Element  Level  Left_Neighbor  Right_Neighbor")
    print("--------------------------------------------")
    
    def find_neighbors(elem, label_mat, active):
        left = find_active_neighbor(elem, 'left', label_mat, active)
        right = find_active_neighbor(elem, 'right', label_mat, active)
        return left, right
        
    for elem in sorted(active):
        level = label_mat[elem-1][4]
        left, right = find_neighbors(elem, label_mat, active)
        print(f"{elem:7d}  {level:5d}  {left if left else 'None':13}  {right if right else 'None':13}")




# def enforce_2_1_balance(label_mat, active, marks):
#     """
#     Enforces 2:1 balance by propagating refinement as needed.
#     Uses level information directly from label_mat[:,4].
    
#     Args:
#         label_mat: Element family relationships [elem, parent, child1, child2, level]
#         active: Currently active elements 
#         marks: Current refinement marks (-1: coarsen, 0: no change, 1: refine)
        
#     Returns:
#         Updated marks array ensuring 2:1 balance
#     """
#     from collections import deque
    
#     # Keep track of cells we've processed
#     processed = set()
    
#     # Process queue
#     queue = deque()
#     for i, mark in enumerate(marks):
#         if mark == 1:  # Initially add all refinement marks
#             queue.append(i)
            
#     while queue:
#         idx = queue.popleft()
#         if idx in processed:
#             continue
            
#         elem = active[idx]
#         elem_level = label_mat[elem-1][4]  # Get level directly from label_mat
        
#         # Get neighbors
#         neighbors = get_element_neighbors(elem, label_mat, active)
        
#         # Check each neighbor
#         for neighbor in neighbors:
#             if neighbor is None:
#                 continue
                
#             neighbor_idx = np.where(active == neighbor)[0][0]
#             neighbor_level = label_mat[neighbor-1][4]  # Get level from label_mat
            
#             # If neighbor would be more than 1 level coarser after refinement
#             if elem_level + 1 - neighbor_level > 1:
#                 # Mark neighbor for refinement
#                 marks[neighbor_idx] = 1
#                 queue.append(neighbor_idx)
                
#         processed.add(idx)
        
#     # Now check if any coarsening would violate 2:1 balance
#     for i, mark in enumerate(marks):
#         if mark == -1:
#             elem = active[i]
#             elem_level = label_mat[elem-1][4]
            
#             # Check if this element can be coarsened
#             parent = label_mat[elem-1][1]
#             if parent == 0:  # Can't coarsen root elements
#                 marks[i] = 0
#                 continue
                
#             # Find sibling
#             siblings = []
#             if elem > 1 and label_mat[elem-2][1] == parent:
#                 siblings.append(elem-1)
#             if elem < len(label_mat) and label_mat[elem][1] == parent:
#                 siblings.append(elem+1)
            
#             # Both siblings must be marked for coarsening
#             sibling_marked = all(
#                 s in active and marks[np.where(active == s)[0][0]] == -1 
#                 for s in siblings
#             )
#             if not sibling_marked:
#                 marks[i] = 0
#                 continue
                
#             # Check neighbors of both this element and siblings
#             all_neighbors = []
#             all_neighbors.extend(get_element_neighbors(elem, label_mat, active))
#             for sib in siblings:
#                 if sib in active:
#                     all_neighbors.extend(get_element_neighbors(sib, label_mat, active))
            
#             # Remove duplicates and None values
#             all_neighbors = [n for n in set(all_neighbors) if n is not None]
            
#             # Check if coarsening would violate 2:1 balance with any neighbor
#             for neighbor in all_neighbors:
#                 neighbor_level = label_mat[neighbor-1][4]
                
#                 # If neighbor is too refined relative to coarsened element
#                 if neighbor_level - (elem_level - 1) > 1:
#                     marks[i] = 0  # Prevent coarsening
#                     break
                    
#     return marks





# def enforce_balance(active, label_mat, cur_grid, info_mat, nop, cur_coords, PS1, PS2, PG1, PG2, ngl, xgl, qp):
#     """
#     Single step 2:1 Balance enforcement. This routine handles balance marking, mesh adapting, and solution adapting. THIS NEEDS TO BE CALLED IN A LOOP
#     """
    
#     bal_marks = balance_mark(active, label_mat)
#     pre_active = active  
#     pre_grid = cur_grid
#     pre_coord = cur_coords

#     bal_grid, bal_active, ref_marks, bal_nelem, npoin_cg, bal_npoin_dg = adapt_mesh(nop, pre_grid, active, label_mat, info_mat, bal_marks)
#     bal_coord, bal_intma, periodicity = create_grid_us(ngl, bal_nelem, npoin_cg, bal_npoin_dg, xgl, bal_grid)
#     bal_q = adapt_sol(qp, pre_coord, bal_marks, pre_active, label_mat, PS1, PS2, PG1, PG2, ngl)

#     # Update for next level
#     # qp = bal_q
#     # active = bal_active
#     # nelem = bal_nelem
#     # intma = bal_intma
#     # coord = bal_coord
#     # grid = bal_grid
#     # npoin_dg = bal_npoin_dg

#     return bal_q, bal_active, bal_nelem, bal_intma, bal_coord, bal_grid, bal_npoin_dg, periodicity
