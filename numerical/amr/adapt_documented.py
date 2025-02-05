"""
Mesh and Solution Adaptation Module

This module implements mesh adaptation capabilities including:
- Element refinement and coarsening
- Solution projection between refined/coarsened elements
- Preservation of solution accuracy during mesh modification
"""

import numpy as np

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
    Unified mesh adaptation routine handling both refinement and derefinement.
    
    This function:
    1. Processes refinement/coarsening marks one at a time
    2. Updates grid coordinates appropriately
    3. Maintains correct element relationships
    4. Updates active element tracking
    
    Args:
        nop (int): Polynomial order
        cur_grid (array): Current grid node coordinates
        active (array): Currently active element indices
        label_mat (array): Element family relationships [elem_id, parent_id, child1_id, child2_id]
        info_mat (array): Element metadata [elem_id, parent_id, level, left_coord, right_coord]
        marks (array): Refinement markers (-1: coarsen, 0: no change, 1: refine)
    
    Returns:
        tuple: (adapted_grid, new_active, marks, new_nelem, new_npoin_cg, new_npoin_dg)
            adapted_grid: Updated grid coordinates
            new_active: Updated active element indices
            marks: Updated refinement markers
            new_nelem: Number of elements after adaptation
            new_npoin_cg: Number of continuous grid points
            new_npoin_dg: Number of discontinuous grid points
    """
    ngl = nop + 1
    
    # Early exit if no adaptation needed
    if not np.any(marks):
        new_nelem = len(active)
        return (cur_grid, active, marks, new_nelem, 
                nop * new_nelem + 1, ngl * new_nelem)
    
    # Process adaptations sequentially
    i = 0
    while i < len(marks):
        if marks[i] == 0:  # No change needed
            i += 1
            continue
            
        if marks[i] > 0:  # Handle refinement
            elem = active[i]
            parent_idx = elem - 1
            c1, c2 = label_mat[parent_idx][2:4]  # Get children IDs
            c1_r = info_mat[c1-1][4]  # Get right boundary of first child
            
            # Update grid by inserting new point
            cur_grid = np.insert(cur_grid, i + 1, c1_r)
            
            # Update active elements - replace parent with children
            active = np.concatenate([
                active[:i],
                [c1, c2],
                active[i+1:]
            ])
            
            # Reset marks for new elements
            marks = np.concatenate([
                marks[:i],
                [0, 0],
                marks[i+1:]
            ])
            
            i += 2  # Skip past newly added element
            
        else:  # Handle coarsening (marks[i] < 0)
            elem = active[i]
            parent = label_mat[elem-1][1]
            
            # Find sibling element
            if label_mat[elem-2][1] == parent and i > 0 and marks[i-1] < 0:
                sib_idx = i - 1  # Sibling is previous element
                min_idx = sib_idx
            elif i + 1 < len(marks) and label_mat[elem][1] == parent and marks[i+1] < 0:
                sib_idx = i + 1  # Sibling is next element
                min_idx = i
            else:
                # No valid sibling for coarsening
                i += 1
                continue
                
            # Remove mid-point between coarsened elements
            cur_grid = np.delete(cur_grid, min_idx + 1)
            
            # Update active elements - replace children with parent
            active = np.concatenate([
                active[:min_idx],
                [parent],
                active[min_idx+2:]
            ])
            
            # Update marks
            marks = np.concatenate([
                marks[:min_idx],
                [0],
                marks[min_idx+2:]
            ])
            
            i = min_idx + 1
    
    # Calculate new mesh dimensions
    new_nelem = len(active)
    new_npoin_cg = nop * new_nelem + 1  # Continuous grid points
    new_npoin_dg = ngl * new_nelem      # Discontinuous grid points
    
    return cur_grid, active, marks, new_nelem, new_npoin_cg, new_npoin_dg

def adapt_sol(q, coord, marks, active, label_mat, PS1, PS2, PG1, PG2, ngl):
    """
    Adapts solution values during mesh adaptation using projection operations.
    
    This function handles:
    1. Solution projection during element refinement (parent -> children)
    2. Solution gathering during element coarsening (children -> parent)
    3. Maintaining solution continuity and accuracy
    
    Args:
        q (array): Current solution values
        coord (array): Grid point coordinates
        marks (array): Original refinement markers (-1: coarsen, 0: no change, 1: refine)
        active (array): Original active element indices (pre-adaptation)
        label_mat (array): Element family relationships [elem_id, parent_id, child1_id, child2_id]
        PS1, PS2 (array): Scatter matrices for child 1 and 2 [ngl, ngl]
        PG1, PG2 (array): Gather matrices for child 1 and 2 [ngl, ngl]
        ngl (int): Number of LGL points per element
        
    Returns:
        array: Solution values projected onto adapted mesh
    """
    new_q = []
    
    i = 0
    while i < len(marks):
        if marks[i] == 0:  # No adaptation
            elem_vals = q[i*ngl:(i+1)*ngl]
            new_q.extend(elem_vals)
            i += 1
            
        elif marks[i] == 1:  # Refinement
            parent_elem = active[i]
            parent_vals = q[i*ngl:(i+1)*ngl]
            
            # Get children from label matrix
            child1, child2 = label_mat[parent_elem-1][2:4]
            
            # Project parent solution to children using scatter matrices
            child1_vals = PS1 @ parent_vals
            child2_vals = PS2 @ parent_vals
            
            # Store both children's solutions
            new_q.extend(child1_vals)
            new_q.extend(child2_vals)
            i += 1
            
        else:  # Coarsening (marks[i] == -1)
            if i + 1 < len(marks) and marks[i+1] == -1:  # Valid coarsening pair
                # Get elements being coarsened
                child1_elem = active[i]
                child2_elem = active[i+1]
                
                # Get parent element
                parent = label_mat[child1_elem-1][1]
                
                # Get values from both children
                child1_vals = q[i*ngl:(i+1)*ngl]
                child2_vals = q[(i+1)*ngl:(i+2)*ngl]
                
                # Project children solutions to parent using gather matrices
                parent_vals = PG1 @ child1_vals + PG2 @ child2_vals
                
                # Store parent solution
                new_q.extend(parent_vals)
                
                i += 2  # Skip both coarsened elements
            else:
                # Unpaired coarsening mark - maintain current solution
                elem_vals = q[i*ngl:(i+1)*ngl]
                new_q.extend(elem_vals)
                i += 1
    
    return np.array(new_q)