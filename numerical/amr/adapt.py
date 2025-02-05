import numpy as np


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


