import numpy as np


#~~~~~~~~~~~~~~~~~~~ single refine/coarsen routine ~~~~~~~~~~~~~~~~~~~~~~~~

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

def get_element_neighbors(elem, label_mat, active):
    """
    Gets neighboring elements by checking siblings and parent/child relationships.
    
    Returns list of neighbor element numbers or None for boundaries.
    """
    neighbors = []
    
    # Check previous element
    if elem > 1 and (elem-1) in active:
        neighbors.append(elem-1)
        
    # Check next element  
    if elem < len(label_mat) and (elem+1) in active:
        neighbors.append(elem+1)
        
    return neighbors

def enforce_2_1_balance(label_mat, info_mat, active, marks):
    """
    Enforces 2:1 balance by propagating refinement as needed.
    
    Args:
        label_mat: Element family relationships [elem, parent, child1, child2]
        info_mat: Element information including level
        active: Currently active elements 
        marks: Current refinement marks (-1: coarsen, 0: no change, 1: refine)
        
    Returns:
        Updated marks array ensuring 2:1 balance
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
        elem_level = info_mat[elem-1][2]  # Current element's level
        
        # Get neighbors
        neighbors = get_element_neighbors(elem, label_mat, active)
        
        # Check each neighbor
        for neighbor in neighbors:
            if neighbor is None:
                continue
                
            neighbor_idx = np.where(active == neighbor)[0][0]
            neighbor_level = info_mat[neighbor-1][2]
            
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
            elem_level = info_mat[elem-1][2]
            
            neighbors = get_element_neighbors(elem, label_mat, active)
            
            for neighbor in neighbors:
                if neighbor is None:
                    continue
                    
                neighbor_idx = np.where(active == neighbor)[0][0]
                neighbor_level = info_mat[neighbor-1][2]
                
                # If neighbor is too refined relative to coarsened element
                if neighbor_level - (elem_level - 1) > 1:
                    marks[i] = 0  # Prevent coarsening
                    break
                    
    return marks




#The following is the same routine with debugging comments
# def adapt_sol(q, coord, marks, active, label_mat, PS1, PS2, PG1, PG2, ngl):
#     """
#     Adapts solution values during mesh adaptation using scatter/gather operations.
    
#     Args:
#         q (array): Current solution values
#         marks (array): Original refinement markers (-1: coarsen, 0: no change, 1: refine)
#         active (array): Original (pre-refinement) active element indices. Must correspond 
#                        to the original mesh that marks refers to, not the adapted mesh.
#         label_mat (array): Element family relationships [elem, parent, child1, child2]
#         PS1, PS2 (array): Scatter matrices for child 1 and 2 [ngl, ngl]
#         PG1, PG2 (array): Gather matrices for child 1 and 2 [ngl, ngl]
#         ngl (int): Number of LGL points per element
        
#     Returns:
#         array: Adapted solution values
#     """
#     # print("\nStarting solution adaptation:")
#     # print(f"Initial q shape: {q.shape}")
#     # print(f"Original (pre-refinement) active elements: {active}")
#     # print(f"Original marks: {marks}")
    
#     new_q = []
    
#     i = 0
#     while i < len(marks):
#         # print(f"\nProcessing mark {i} for original active element {active[i]}:")
#         # print(f"Mark value: {marks[i]}")
        
#         if marks[i] == 0:
#             # No adaptation - copy solution values
#             elem_vals = q[i*ngl:(i+1)*ngl]
#             # print(f"No adaptation - copying values for element {active[i]}")
#             new_q.extend(elem_vals)
#             i += 1
            
#         elif marks[i] == 1:
#             # Refinement - scatter parent solution to children
#             parent_elem = active[i]
#             parent_vals = q[i*ngl:(i+1)*ngl]

#             # print(f"\nRefinement Debug for position {i}:")
#             # print(f"Parent element number in tree: {parent_elem}")
#             # print(f"Current active elements: {active}")

#             # print(f"\nProjection Debug for element {parent_elem}:")
#             # print(f"Element coordinates: {coord[i*ngl:(i+1)*ngl]}")
#             # print(f"Parent values: {parent_vals}")
#             # print(f"Using PS1:\n{PS1}")
#             # print(f"Using PS2:\n{PS2}")
            
#             # Get children from label_mat using original element number
#             child1, child2 = label_mat[parent_elem-1][2:4]
            
#             # print(f"Refining original element {parent_elem} into children {child1}, {child2}")
#             # print(f"Parent values: {parent_vals}")
            
#             # Scatter to get child solutions using mesh-independent matrices
#             child1_vals = PS1 @ parent_vals
#             child2_vals = PS2 @ parent_vals
            
#             # print(f"Child 1 values: {child1_vals}")
#             # print(f"Child 2 values: {child2_vals}")
            
#             # Add both children's solutions
#             new_q.extend(child1_vals)
#             new_q.extend(child2_vals)
#             i += 1
            
#         else:  # marks[i] == -1
#             # Handle coarsening
#             if i + 1 < len(marks) and marks[i+1] == -1:
#                 # Get the original elements we're coarsening
#                 child1_elem = active[i]
#                 child2_elem = active[i+1]
                
#                 # Get parent from label_mat using original element number
#                 parent = label_mat[child1_elem-1][1]  # Both children have same parent

#                 # print(f"\nCoarsening Debug for position {i}:")
#                 # print(f"Child elements being coarsened: {child1_elem}, {child2_elem}")
#                 # print(f"Current active elements: {active}")
                
#                 # print(f"Coarsening original children {child1_elem}, {child2_elem} back to parent {parent}")
                
#                 # Get values for both children
#                 child1_vals = q[i*ngl:(i+1)*ngl]
#                 child2_vals = q[(i+1)*ngl:(i+2)*ngl]
                
#                 # Gather children solutions to parent using mesh-independent matrices
#                 parent_vals = PG1 @ child1_vals + PG2 @ child2_vals
                
#                 # print(f"Gathered parent values: {parent_vals}")
                
#                 # Add parent solution
#                 new_q.extend(parent_vals)
                
#                 # Skip both coarsening marks
#                 i += 2
#             else:
#                 # print(f"Warning: Unpaired coarsening mark at original element {active[i]}")
#                 elem_vals = q[i*ngl:(i+1)*ngl]
#                 new_q.extend(elem_vals)
#                 i += 1
    
#     result = np.array(new_q)
#     # print(f"\nFinal adapted solution shape: {result.shape}")
#     return result



# def adapt_sol(q, coord, marks, active, label_mat, PS1, PS2, PG1, PG2, ngl):
#     """
#     Adapts solution values during mesh adaptation using scatter/gather operations.
    
#     Args:
#         q (array): Current solution values
#         marks (array): Original refinement markers (-1: coarsen, 0: no change, 1: refine)
#         active (array): Original (pre-refinement) active element indices. Must correspond 
#                        to the original mesh that marks refers to, not the adapted mesh.
#         label_mat (array): Element family relationships [elem, parent, child1, child2]
#         PS1, PS2 (array): Scatter matrices for child 1 and 2 [nelem, ngl, ngl]
#         PG1, PG2 (array): Gather matrices for child 1 and 2 [nelem, ngl, ngl]
#         ngl (int): Number of LGL points per element
        
#     Returns:
#         array: Adapted solution values
        
#     Notes:
#         - The active array must be the original active array before any refinement/coarsening,
#           matching the structure that the marks array refers to.
#         - For refinement (marks[i]=1), element active[i] will be split into its children
#           from label_mat
#         - For coarsening (marks[i]=marks[i+1]=-1), elements active[i] and active[i+1]
#           will be combined into their parent from label_mat
#     """
#     print("\nStarting solution adaptation:")
#     print(f"Initial q shape: {q.shape}")
#     print(f"Original (pre-refinement) active elements: {active}")
#     print(f"Original marks: {marks}")
    
#     new_q = []
#     nelem = PS1.shape[0]
    
#     i = 0
#     while i < len(marks):
#         print(f"\nProcessing mark {i} for original active element {active[i]}:")
#         print(f"Mark value: {marks[i]}")
        
#         if marks[i] == 0:
#             # No adaptation - copy solution values
#             elem_vals = q[i*ngl:(i+1)*ngl]
#             print(f"No adaptation - copying values for element {active[i]}")
#             new_q.extend(elem_vals)
#             i += 1
            
#         elif marks[i] == 1:
#             # Refinement - scatter parent solution to children
#             # parent_elem = active[i]
#             # matrix_idx = (parent_elem - 1) % nelem

#             # For refinement:
#             parent_elem = active[i]
#             matrix_idx = i  # Use position in pre-refined mesh
#             parent_vals = q[i*ngl:(i+1)*ngl]

            

#             print(f"\nRefinement Debug for position {i}:")
#             print(f"Parent element number in tree: {parent_elem}")
#             print(f"Using projection matrix {i}")
#             print(f"Current active elements: {active}")

#             print(f"\nProjection Debug for element {parent_elem}:")
#             print(f"Element coordinates: {coord[i*ngl:(i+1)*ngl]}")
#             print(f"Parent values: {parent_vals}")
#             print(f"Using PS1[{matrix_idx}]:\n{PS1[matrix_idx]}")
#             print(f"Using PS2[{matrix_idx}]:\n{PS2[matrix_idx]}")









            
#             # Get children from label_mat using original element number
#             child1, child2 = label_mat[parent_elem-1][2:4]
#             parent_vals = q[i*ngl:(i+1)*ngl]
            
#             print(f"Refining original element {parent_elem} into children {child1}, {child2}")
#             print(f"Using matrix index: {matrix_idx}")
#             print(f"Parent values: {parent_vals}")
            
#             # Scatter to get child solutions
#             child1_vals = PS1[matrix_idx] @ parent_vals
#             child2_vals = PS2[matrix_idx] @ parent_vals
            
#             print(f"Child 1 values: {child1_vals}")
#             print(f"Child 2 values: {child2_vals}")
            
#             # Add both children's solutions
#             new_q.extend(child1_vals)
#             new_q.extend(child2_vals)
#             i += 1
            
#         else:  # marks[i] == -1
#             # Handle coarsening
#             if i + 1 < len(marks) and marks[i+1] == -1:
#                 # Get the original elements we're coarsening
#                 child1_elem = active[i]
#                 child2_elem = active[i+1]
                
#                 # Get parent from label_mat using original element number
#                 parent = label_mat[child1_elem-1][1]  # Both children have same parent
#                 # matrix_idx = (parent - 1) % nelem

#                 # For coarsening:
#                 matrix_idx = i  # Use position of first child in pre-refined mesh

#                 print(f"\nCoarsening Debug for position {i}:")
#                 print(f"Child elements being coarsened: {child1_elem}, {child2_elem}")
#                 print(f"Using projection matrices at position {i}")
#                 print(f"Current active elements: {active}")

                
                
#                 print(f"Coarsening original children {child1_elem}, {child2_elem} back to parent {parent}")
#                 print(f"Using matrix index: {matrix_idx}")
                
#                 # Get values for both children
#                 child1_vals = q[i*ngl:(i+1)*ngl]
#                 child2_vals = q[(i+1)*ngl:(i+2)*ngl]
                
#                 # print(f"Child 1 values: {child1_vals}")
#                 # print(f"Child 2 values: {child2_vals}")
                
#                 # # Gather children solutions to parent
#                 # parent_vals = PG1[matrix_idx] @ child1_vals + PG2[matrix_idx] @ child2_vals

#                 # Gather children solutions to parent using position i
#                 parent_vals = PG1[i] @ child1_vals + PG2[i] @ child2_vals
                
#                 print(f"Gathered parent values: {parent_vals}")
                
#                 # Add parent solution
#                 new_q.extend(parent_vals)
                
#                 # Skip both coarsening marks
#                 i += 2
#             else:
#                 print(f"Warning: Unpaired coarsening mark at original element {active[i]}")
#                 elem_vals = q[i*ngl:(i+1)*ngl]
#                 new_q.extend(elem_vals)
#                 i += 1
    
#     result = np.array(new_q)
#     print(f"\nFinal adapted solution shape: {result.shape}")
#     return result


# def adapt_sol(q, marks, active, label_mat, PS1, PS2, PG1, PG2, ngl):
#     """
#     Adapts solution values during mesh adaptation using scatter/gather operations.
#     Works with original marks array indicating which parent elements need adaptation.
    
#     Args:
#         q (array): Current solution values on original mesh
#         marks (array): Original refinement markers (-1: coarsen, 0: no change, 1: refine)
#         active (array): Original active element indices
#         label_mat (array): Element family relationships [rows, 4]
#         PS1, PS2 (array): Scatter matrices for child 1 and 2 [nelem, ngl, ngl]
#         PG1, PG2 (array): Gather matrices for child 1 and 2 [nelem, ngl, ngl]
#         ngl (int): Number of LGL points per element
        
#     Returns:
#         array: Adapted solution values on new mesh
#     """
#     print("\nStarting solution adaptation:")
#     print(f"Initial q shape: {q.shape}")
#     print(f"Original active elements: {active}")
#     print(f"Original marks: {marks}")
#     print(f"PS1 shape: {PS1.shape}, PS2 shape: {PS2.shape}")
    
#     new_q = []
#     nelem = PS1.shape[0]
    
#     for i in range(len(marks)):
#         print(f"\nProcessing element {i}:")
#         print(f"Mark value: {marks[i]}")
        
#         if marks[i] == 0:
#             # No adaptation - copy solution values
#             elem_vals = q[i*ngl:(i+1)*ngl]
#             print(f"No adaptation - copying values: {elem_vals}")
#             new_q.extend(elem_vals)
            
#         elif marks[i] == 1:
#             # Refinement - scatter parent solution to children
#             elem = active[i]
#             matrix_idx = (elem - 1) % nelem
#             parent_vals = q[i*ngl:(i+1)*ngl]
            
#             print(f"Refinement for element {elem} (matrix index {matrix_idx})")
#             print(f"Parent values: {parent_vals}")
            
#             # Scatter to get child solutions
#             child1_vals = PS1[matrix_idx] @ parent_vals
#             child2_vals = PS2[matrix_idx] @ parent_vals
            
#             print(f"Child 1 values: {child1_vals}")
#             print(f"Child 2 values: {child2_vals}")
            
#             # Add both children's solutions
#             new_q.extend(child1_vals)
#             new_q.extend(child2_vals)
            
#         else:  # marks[i] == -1
#             # Should handle coarsening if needed
#             print(f"Warning: Coarsening not implemented for mark {i}")
#             elem_vals = q[i*ngl:(i+1)*ngl]
#             new_q.extend(elem_vals)
    
#     result = np.array(new_q)
#     print(f"\nFinal adapted solution shape: {result.shape}")
#     return result


# def adapt_sol(q, marks, active, label_mat, PS1, PS2, PG1, PG2, ngl):
#     """
#     Adapts solution values during mesh adaptation using scatter/gather operations.
    
#     Args:
#         q (array): Current solution values
#         marks (array): Refinement markers (-1: coarsen, 0: no change, 1: refine)
#         active (array): Active element indices
#         label_mat (array): Element family relationships [rows, 4]
#         PS1, PS2 (array): Scatter matrices for child 1 and 2
#         PG1, PG2 (array): Gather matrices for child 1 and 2
#         ngl (int): Number of LGL points per element
        
#     Returns:
#         array: Adapted solution values
#     """
#     new_q = []
#     i = 0
    
#     while i < len(marks):
#         if marks[i] == 0:
#             # No change - copy solution values for this element
#             elem_vals = q[i*ngl:(i+1)*ngl]
#             new_q.extend(elem_vals)
#             i += 1
            
#         elif marks[i] > 0:
#             # Refinement - scatter parent solution to children
#             elem = active[i]
#             parent_idx = elem - 1
            
#             # Get parent solution values
#             parent_vals = q[i*ngl:(i+1)*ngl]
            
#             # Scatter to get child solutions
#             child1_vals = PS1[parent_idx] @ parent_vals
#             child2_vals = PS2[parent_idx] @ parent_vals
            
#             # Add both children's solutions
#             new_q.extend(child1_vals)
#             new_q.extend(child2_vals)
            
#             i += 2  # Skip past the refined element
            
#         else:  # marks[i] < 0
#             # Coarsening - gather children solutions to parent
#             elem = active[i]
#             parent = label_mat[elem-1][1]
            
#             # Find sibling
#             if i > 0 and label_mat[elem-2][1] == parent and marks[i-1] < 0:
#                 # Previous element is sibling
#                 sib_idx = i - 1
#                 child1_vals = q[sib_idx*ngl:(sib_idx+1)*ngl]
#                 child2_vals = q[i*ngl:(i+1)*ngl]
                
#                 # Gather children solutions to parent
#                 parent_vals = (PG1[parent-1] @ child1_vals + 
#                              PG2[parent-1] @ child2_vals)
                
#                 # Replace previous solution (sibling) with parent
#                 new_q = new_q[:-ngl]  # Remove sibling solution
#                 new_q.extend(parent_vals)
                
#                 i += 2  # Skip past both coarsened elements
                
#             elif (i + 1 < len(marks) and 
#                   label_mat[elem][1] == parent and marks[i+1] < 0):
#                 # Next element is sibling - skip and handle in next iteration
#                 i += 1
#                 continue
#             else:
#                 # No valid sibling - copy current solution
#                 elem_vals = q[i*ngl:(i+1)*ngl]
#                 new_q.extend(elem_vals)
#                 i += 1
    
#     return np.array(new_q)