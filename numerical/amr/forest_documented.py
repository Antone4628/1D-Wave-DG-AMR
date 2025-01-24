"""
Hierarchical Mesh Refinement Module

This module implements hierarchical mesh refinement capabilities for adaptive mesh refinement (AMR).
Key features:
- Binary tree structure for element refinement
- Parent-child relationship tracking
- Element marking based on solution criteria
"""

import numpy as np

def next_level(xelem):
    """
    Creates next refinement level by inserting midpoints between existing nodes.
    
    Args:
        xelem (array): Element boundary coordinates at current level
    Returns:
        array: New grid coordinates with midpoints added
    """
    m = len(xelem)
    out = np.zeros((2*m-1), dtype=xelem.dtype)
    out[::2] = xelem  # Copy existing points to even indices
    midpoints = (xelem[:-1] + xelem[1:]) / 2  # Calculate midpoints
    out[1::2] = midpoints  # Insert midpoints at odd indices
    return out

def level_arrays(xelem, max_level):
    """
    Generates coordinate arrays for all refinement levels.
    
    Args:
        xelem (array): Initial (level 0) grid coordinates
        max_level (int): Maximum refinement level
    Returns:
        list: Arrays of node coordinates for each level [level_0, level_1, ..., level_max]
    """
    levels = [xelem]  # Start with initial grid
    for i in range(max_level):
        next_lev = next_level(levels[i])  # Generate next level
        levels.append(next_lev)
    return levels

def stacker(level):
    """
    Creates pairs of element boundaries for refinement bookkeeping.
    
    Args:
        level (array): Grid coordinates at current level
    Returns:
        array: Element boundary pairs shaped as [num_elements, 2]
    """
    m = int(len(level))
    out = np.zeros((2*m-1))
    out[::2] = level  # Copy points to even indices
    out[1::2] = out[2::2]  # Copy right boundaries to odd indices
    stacker = out[:-1].reshape(int(m-1), 2)  # Reshape into pairs
    return stacker

def vstacker(levels):
    """
    Combines element boundary pairs from all refinement levels.
    
    Args:
        levels (list): List of coordinate arrays for each level
    Returns:
        array: Vertically stacked element pairs from all levels
    """
    stacks = [stacker(level) for level in levels]
    vstack = np.vstack(stacks)
    return vstack

def forest(xelem0, max_level):
    """
    Creates hierarchical mesh structure for AMR operations.
    
    This function builds the data structures needed to track element refinement:
    - Label matrix: Stores parent-child relationships
    - Info matrix: Stores element metadata and coordinates
    - Active grid: Tracks currently active (leaf) elements
    
    Args:
        xelem0 (array): Initial grid coordinates
        max_level (int): Maximum allowed refinement level
    Returns:
        tuple: (label_mat, info_mat, active_grid)
            label_mat: [num_total_elements, 4] array storing:
                      [element_id, parent_id, child1_id, child2_id]
            info_mat: [num_total_elements, 5] array storing:
                     [element_id, parent_id, level, left_coord, right_coord]
            active_grid: Array of currently active element IDs
    """
    levels = max_level + 1
    elems0 = len(xelem0) - 1  # Initial number of elements
    rows = elems0  # Start with base elements
    lmt = 0  # Last element that can have children
    
    # Calculate total elements across all levels
    elems = np.zeros(levels, dtype=int)
    elems[0] = elems0
    for i in range(levels-1):
        a = 2**(i+1) * elems0
        rows += a
        lmt = rows - a
    
    # Initialize matrices
    label_mat = np.zeros([rows, 4], dtype=int)  # Element relationships
    info_mat = np.zeros([rows, 3])  # Element metadata
    
    # Track elements per level for level calculation
    elems_per_level = [elems0]
    for i in range(max_level):
        elems_per_level.append(elems_per_level[-1] * 2)
    
    # Fill matrices
    ctr = 2
    for j in range(rows):
        div = elems0
        
        # Set element and parent IDs
        label_mat[j][0] = j + 1
        info_mat[j][0] = j + 1
        
        if j < div:  # Base level elements
            label_mat[j][1] = j//div
            info_mat[j][1] = int(j//div)
        else:  # Refined elements
            label_mat[j][1] = (ctr)//2
            info_mat[j][1] = int(ctr//2)
            ctr += 1
            
        # Set children IDs for non-leaf elements
        if j < lmt:
            label_mat[j][2] = div + (2*j+1)  # Left child
            label_mat[j][3] = div + 2*(j+1)  # Right child
        
        # Calculate element's refinement level
        cum_sum = 0
        for lvl, num_elems in enumerate(elems_per_level):
            if j < (cum_sum + num_elems):
                info_mat[j][2] = lvl
                break
            cum_sum += num_elems
    
    # Add coordinate information
    levels_arr = level_arrays(xelem0, max_level)
    vstack = vstacker(levels_arr)
    coord_mat = np.hstack((info_mat, vstack))
    
    # Initialize active grid with base elements
    active_grid = np.arange(1, len(xelem0))
    
    return label_mat, coord_mat, active_grid

def mark(active_grid, label_mat, intma, q, criterion):
    """
    Marks elements for refinement/coarsening based on solution properties.
    
    For criterion = 1 (default):
    - Elements are marked for refinement if max(solution) >= 0.5
    - Elements are marked for coarsening if max(solution) < 0.5
    - Coarsening only occurs if both sibling elements qualify
    
    Args:
        active_grid (array): Currently active element IDs
        label_mat (array): Element relationship matrix [elem_id, parent, child1, child2]
        intma (array): Element-node connectivity matrix
        q (array): Solution values at nodes
        criterion (int): Selection of marking criteria
            1: Default threshold-based marking
            2: Reserved for custom criteria
    
    Returns:
        array: Element markers (-1: coarsen, 0: no change, 1: refine)
    """
    n_active = len(active_grid)
    marks = np.zeros(n_active, dtype=int)
    refs = []  # Elements to refine
    defs = []  # Elements to coarsen
    
    # Get parent and child info for active elements
    parents = label_mat[active_grid - 1, 1]
    children = label_mat[active_grid - 1, 2:4]
    
    # Process each active element
    for idx, (elem, parent) in enumerate(zip(active_grid, parents)):
        # Get solution values in element
        elem_nodes = intma[:, idx]
        elem_sols = q[elem_nodes]
        max_sol = np.max(elem_sols)
        
        if criterion == 1:
            # Check refinement criteria
            if max_sol >= 0.5 and children[idx, 0] != 0:
                refs.append(elem)
                marks[idx] = 1
                continue
                
            # Check coarsening criteria
            if max_sol < 0.5 and parent != 0:
                # Find sibling element
                sibling = None
                if elem > 1 and label_mat[elem-2, 1] == parent:
                    sibling = elem - 1
                    sib_idx = idx - 1
                elif elem < len(label_mat) and label_mat[elem, 1] == parent:
                    sibling = elem + 1
                    sib_idx = idx + 1
                    
                # Check if sibling also qualifies for coarsening
                if sibling in active_grid:
                    sib_nodes = intma[:, sib_idx]
                    sib_sols = q[sib_nodes]
                    
                    if np.max(sib_sols) < 0.5 and sibling not in defs:
                        marks[idx] = marks[sib_idx] = -1
                        defs.extend([elem, sibling])
                        
        elif criterion == 2:
            # Reserved for additional marking criteria
            pass
    
    return marks

def elem_info(elem, label_mat):
    """
    Prints family relationship information for a specific element.
    
    Args:
        elem (int): Element ID to query
        label_mat (array): Element relationship matrix
    """
    parent = label_mat[elem-1][1]
    c1 = label_mat[elem-1][2]
    c2 = label_mat[elem-1][3]
    print(f'\n\n element number {elem} has parent {parent} and children {c1} and {c2}')
    
    if parent != 0:
        # Find and print sibling
        if label_mat[elem-2][1] == parent:
            sib = elem-1
        elif label_mat[elem][1] == parent:
            sib = elem+1
        print(f'element {elem} has sibling {sib}')