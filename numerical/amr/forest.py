import numpy as np

def next_level(xelem):
    """
    Generates next refinement level by adding midpoints.
    
    Args:
        xelem (array): Element boundary coordinates
    Returns:
        array: Refined grid coordinates including midpoints
    """
    m = len(xelem)
    out = np.zeros((2*m-1),dtype=xelem.dtype)
    out[::2] = xelem
    midpoints = (xelem[:-1] + xelem[1:]) / 2
    out[1::2]=midpoints
    return out

def level_arrays(xelem, max_level):
    """
    Generates all refinement levels up to max_level.
    
    Args:
        xelem (array): Initial grid coordinates
        max_level (int): Maximum refinement level
    Returns:
        list: Arrays of coordinates for each level
    """
    levels = []
    levels.append(xelem)
    for i in range(max_level):
        next_lev = next_level(levels[i])
        levels.append(next_lev)

    return levels

def stacker(level):
    """
    Creates element pairs for a given level.
    
    Args:
        level (array): Grid coordinates at refinement level
    Returns:
        array: Paired element boundaries [m-1,2]
    """
    m = int(len(level))
    # out = np.zeros((2*m-1),dtype=xelem.dtype)
    out = np.zeros((2*m-1))
    out[::2] = level
    out[1::2]= out[2::2]
    stacker = out[:-1].reshape(int(m-1),2)
    return stacker

def vstacker(levels):
    """
    Stacks element pairs from all levels vertically.
    
    Args:
        levels (list): List of grid coordinates at each level
    Returns:
        array: Vertically stacked element pairs
    """
    stacks=[]
    for level in levels:
        stacks.append(stacker(level))
    vstack = np.vstack(stacks)
    return vstack



def forest(xelem0, max_level):
    """
    Creates hierarchical mesh structure for AMR.
    
    Args:
        xelem0 (array): Initial grid coordinates
        max_level (int): Maximum refinement level
    Returns:
        tuple: (label_mat, info_mat, active_grid)
            label_mat: Parent-child relationships
            info_mat: Element information
            active_grid: Currently active elements
    """

    # Initial checks and setup remain the same
    levels = max_level + 1
    elems0 = len(xelem0) - 1
    rows = elems0
    lmt = 0
    elems = np.zeros(levels, dtype=int)
    elems[0] = len(xelem0) - 1
    
    # Calculate total rows needed
    for i in range(levels-1):
        a = 2**(i+1) * elems0
        rows += a
        lmt = rows - a
    
    cols = 4
    label_mat = np.zeros([rows, cols], dtype=int)
    # Modify info_mat to have 3 columns: [element, parent, level]
    info_mat = np.zeros([rows, 3])
    
    # Track how many elements at each level
    elems_per_level = [elems0]
    for i in range(max_level):
        elems_per_level.append(elems_per_level[-1] * 2)
    
    ctr = 2
    cum_elems = 0  # Cumulative elements to track level boundaries
    
    for j in range(rows):
        div = len(xelem0) - 1
        
        # Set element number
        label_mat[j][0] = j + 1
        info_mat[j][0] = j + 1
        
        # Set parent info
        if j < div:
            label_mat[j][1] = j//div
            info_mat[j][1] = int(j//div)
        else:
            label_mat[j][1] = (ctr)//2
            info_mat[j][1] = int(ctr//2)
            ctr += 1
            
        # Set children in label_mat
        if j < lmt:
            label_mat[j][2] = div + (2*j+1)
            label_mat[j][3] = div + 2*(j+1)
        
        # Calculate and set level in info_mat
        cum_sum = 0
        level = 0
        for lvl, num_elems in enumerate(elems_per_level):
            if j < (cum_sum + num_elems):
                level = lvl
                break
            cum_sum += num_elems
        info_mat[j][2] = level
    
    # Create tree-info matrix
    levels_arr = level_arrays(xelem0, max_level)
    vstack = vstacker(levels_arr)
    coord_mat = np.hstack((info_mat, vstack))
    
    # Set up active grid
    active_grid = np.zeros(len(xelem0)-1, dtype=int)
    for k in range(len(active_grid)):
        active_grid[k] = k + 1
        
    return label_mat, coord_mat, active_grid

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



#~~~~~~~~~~~~~~~~~This is the working marks. It is commented out while I worked on adding criterion argument to a copy
# def mark(active_grid, label_mat, intma, q):
#     """
#     Mark elements for refinement/coarsening based on solution criteria.
    
#     Args:
#         active_grid (array): Currently active element indices
#         label_mat (array): Element family relationships [rows, 4]
#         intma (array): Element-node connectivity
#         q (array): Solution values
        
#     Returns:
#         marks (array): Element markers (-1:derefine, 0:no change, 1:refine)
            
#     Notes:
#         - Elements with solution values >= 0.5 are marked for refinement
#         - Elements with solution values < 0.5 are marked for derefinement
#         - Derefinement only occurs if both siblings meet criteria
#     """
#     n_active = len(active_grid)
#     marks = np.zeros(n_active, dtype=int)
#     refs = []
#     defs = []
    
#     # Pre-compute label matrix lookups
#     parents = label_mat[active_grid - 1, 1]
#     children = label_mat[active_grid - 1, 2:4]
    
#     # Process each active element
#     for idx, (elem, parent) in enumerate(zip(active_grid, parents)):
#         # Get element solution values
#         elem_nodes = intma[:, idx]
#         elem_sols = q[elem_nodes]
#         max_sol = np.max(elem_sols)
        
#         # Check refinement criteria
#         if max_sol >= 0.5 and children[idx, 0] != 0:
#         # if max_sol >= - 0.5 and children[idx, 0] != 0:
#             refs.append(elem)
#             marks[idx] = 1
#             continue
            
#         # Check coarsening criteria
#         if max_sol < 0.5 and parent != 0:
#             # Find sibling
#             sibling = None
#             if elem > 1 and label_mat[elem-2, 1] == parent:
#                 sibling = elem - 1
#                 sib_idx = idx - 1
#             elif elem < len(label_mat) and label_mat[elem, 1] == parent:
#                 sibling = elem + 1
#                 sib_idx = idx + 1
                
#             # Verify sibling status
#             if sibling in active_grid:
#                 sib_nodes = intma[:, sib_idx]
#                 sib_sols = q[sib_nodes]
                
#                 # Mark for coarsening if sibling also qualifies
#                 if np.max(sib_sols) < 0.5 and sibling not in defs:
#                     marks[idx] = marks[sib_idx] = -1
#                     defs.extend([elem, sibling])

    
#     return  marks


def elem_info(elem, label_mat):
    parent = label_mat[elem-1][1]
    c1 = label_mat[elem-1][2]
    c2 = label_mat[elem-1][3]
    print(f'\n\n element number {elem} has parent {parent} and children {c1} and {c2}')
    if (parent != 0):
      # find sibling
        if label_mat[elem-2][1] == parent:
            sib = elem-1
        elif label_mat[elem][1] == parent:
            sib = elem+1
        print(f'eleemnt {elem} has sibling {sib}')




# def forest(xelem0, max_level):
#     """
#     Creates hierarchical mesh structure for AMR.
    
#     Args:
#         xelem0 (array): Initial grid coordinates
#         max_level (int): Maximum refinement level
#     Returns:
#         tuple: (label_mat, info_mat, active_grid)
#             label_mat: Parent-child relationships
#             info_mat: Element information
#             active_grid: Currently active elements
#     """
#     # xelem0 is the initial, level 0 grid
#     # max_level is the maximum refinement level

#     # this routine will take in an inital grid and max level value and return a labeling array and
#     # tree info array whos values are used for refinement


#     #check data type of xelem0 --> must be float
#     assert(xelem0.dtype == 'float64'), "grid data type must be float64"


#     #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#     #Creat labeling martix
#     levels=max_level+1
#     elems0 = len(xelem0)-1
#     # print(f'initial number of elements: {elems0}')
#     rows = elems0
#     lmt = 0
#     elems = np.zeros(levels, dtype = int)
#     elems[0]=len(xelem0)-1
#     for i in range(levels-1):
#         a = 2**(i+1)*elems0
#         rows += a
#     lmt = rows - a #elements ater this value have no children
#     cols = 4
#     label_mat = np.zeros([rows, cols], dtype = int)
#     info_mat  = np.zeros([rows, 2])
#     ctr = 2
#     for j in range(rows):
#         div = len(xelem0)-1
#         label_mat[j][0], info_mat[j][0] = j + 1, j + 1
#         if(j<div):
#             label_mat[j][1], info_mat[j][1] = j//div, int(j//div)
#         else:
#             label_mat[j][1], info_mat[j][1] = (ctr)//2, int(ctr//2)
#             ctr+=1
#         if (j<lmt):
#             label_mat[j][2] = div +(2*j+1)
#             label_mat[j][3] = div + 2*(j+1)

#     #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



#     #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#     #create tree-info matrix
#     levels = level_arrays(xelem0, max_level)
#     vstack = vstacker(levels)
#     coord_mat = np.hstack((info_mat, vstack))
# #     print(f'vstack:\n{vstack}')
# #     print(f'shape: {np.shape(vstack)}')
#     active_grid=np.zeros(len(xelem0)-1, dtype = int)
#     for k in range(len(active_grid)):
#         active_grid[k]=k+1

#     return label_mat, coord_mat, active_grid


#~~~~~~~~~~~~~~~~~~~~~~~~~~~` this mark function has been optimized by Cursor ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`
# def mark(active_grid, label_mat, info_mat, coord, intma, cur_level, max_level, q):
#     """
#     Mark elements for refinement/coarsening based on solution criteria.
    
#     Args:
#         active_grid (array): Currently active element indices
#         label_mat (array): Element family relationships [rows, 4]
#         info_mat (array): Element coordinate information
#         coord (array): Grid coordinates
#         intma (array): Element-node connectivity
#         cur_level (int): Current refinement level
#         max_level (int): Maximum allowed refinement level
#         q (array): Solution values
        
#     Returns:
#         tuple: (refs, defs, marks)
#             refs (list): Elements marked for refinement
#             defs (list): Elements marked for derefinement
#             marks (array): Element markers (-1:derefine, 0:no change, 1:refine)
            
#     Notes:
#         - Elements with solution values >= 0.5 are marked for refinement
#         - Elements with solution values < 0.5 are marked for derefinement
#         - Derefinement only occurs if both siblings meet criteria
#     """
#     n_active = len(active_grid)
#     marks = np.zeros(n_active, dtype=int)
#     refs = []
#     defs = []
    
#     # Pre-compute label matrix lookups
#     parents = label_mat[active_grid - 1, 1]
#     children = label_mat[active_grid - 1, 2:4]
    
#     # Process each active element
#     for idx, (elem, parent) in enumerate(zip(active_grid, parents)):
#         # Get element solution values
#         elem_nodes = intma[:, idx]
#         elem_sols = q[elem_nodes]
#         max_sol = np.max(elem_sols)
        
#         # Check refinement criteria
#         if max_sol >= 0.5 and children[idx, 0] != 0:
#             refs.append(elem)
#             marks[idx] = 1
#             continue
            
#         # Check coarsening criteria
#         if max_sol < 0.5 and parent != 0:
#             # Find sibling
#             sibling = None
#             if elem > 1 and label_mat[elem-2, 1] == parent:
#                 sibling = elem - 1
#                 sib_idx = idx - 1
#             elif elem < len(label_mat) and label_mat[elem, 1] == parent:
#                 sibling = elem + 1
#                 sib_idx = idx + 1
                
#             # Verify sibling status
#             if sibling in active_grid:
#                 sib_nodes = intma[:, sib_idx]
#                 sib_sols = q[sib_nodes]
                
#                 # Mark for coarsening if sibling also qualifies
#                 if np.max(sib_sols) < 0.5 and sibling not in defs:
#                     marks[idx] = marks[sib_idx] = -1
#                     defs.extend([elem, sibling])
    
#     return refs, defs, marks

