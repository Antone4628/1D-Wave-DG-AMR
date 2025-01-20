import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches


def forest(xelem0, max_level):
    # xelem0 is the initial, level 0 grid
    # max_level is the maximum refinement level

    # this routine will take in an inital grid and max level value and return a labeling array and
    # tree info array whos values are used for refinement


    #check data type of xelem0 --> must be float
    assert(xelem0.dtype == 'float64'), "grid data type must be float64"


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Creat labeling martix
    levels=max_level+1
    elems0 = len(xelem0)-1
    # print(f'initial number of elements: {elems0}')
    rows = elems0
    lmt = 0
    elems = np.zeros(levels, dtype = int)
    elems[0]=len(xelem0)-1
    for i in range(levels-1):
        a = 2**(i+1)*elems0
        rows += a
    lmt = rows - a #elements ater this value have no children
    cols = 4
    label_mat = np.zeros([rows, cols], dtype = int)
    info_mat  = np.zeros([rows, 2])
    ctr = 2
    for j in range(rows):
        div = len(xelem0)-1
        label_mat[j][0], info_mat[j][0] = j + 1, j + 1
        if(j<div):
            label_mat[j][1], info_mat[j][1] = j//div, int(j//div)
        else:
            label_mat[j][1], info_mat[j][1] = (ctr)//2, int(ctr//2)
            ctr+=1
        if (j<lmt):
            label_mat[j][2] = div +(2*j+1)
            label_mat[j][3] = div + 2*(j+1)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #create tree-info matrix
    levels = level_arrays(xelem0, max_level)
    vstack = vstacker(levels)
    coord_mat = np.hstack((info_mat, vstack))
#     print(f'vstack:\n{vstack}')
#     print(f'shape: {np.shape(vstack)}')
    active_grid=np.zeros(len(xelem)-1, dtype = int)
    for k in range(len(active_grid)):
        active_grid[k]=k+1

    return label_mat, coord_mat, active_grid

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




def mark(active_grid, label_mat, info_mat, coord, intma, cur_level, max_level, q):
    marks = np.zeros(len(active_grid), dtype = int)
    refs = []
    defs = []

    #index through active_grid and examine refinement criteria
    ind = 0
    for e in active_grid:
        # get element family info:
        # elem_info(e,label_mat)
        parent = label_mat[e-1][1]
        child_1 = label_mat[e-1][2]
        child_2 = label_mat[e-1][3]
        # print(f'\n element number {e} has parent {parent} and children {child_1} and {child_2}')

        #access the element
        # print(f'active grid: {active_grid}')
        # print(f'evaluating element {e} for refinement:')
        # print(f'the nodes in element {e} have the following global indices:')
        index = np.where(active_grid == e)[0][0]
        # print(f'index: {index}')
        # print(f'ctr: {ind}')
        # print(f'intma:')
        # display(intma)
        # print(f'index at e:{index}')
        # print(f'intma at index: {intma[:,index]}')


        elem_nodes = intma[:,index]
        # elem_nodes = intma[:,e-1]
        elem_sols = q[elem_nodes]
        # print(f'element {e} has global solution values: {elem_sols}')
        # print(f'element {e} has global x values: {coord[elem_nodes]}')

        #check for refinement:
        if max(elem_sols) >= 0.5 and child_1 != 0:
          # print(f'element {e}, with index {index} is marked for refinement.')
          refs.append(e)
          marks[index] = 1

        #check for derefinement: Only derefine if previously refined. Else, do nothing.
        elif(max(elem_sols) < 0.5 and parent != 0):
          # Identify sibling and check it's status.
          # print(f'element {e} marked for potential derefinement')
          if label_mat[e-2][1] == parent:
            sib = e-1
          elif label_mat[e][1] == parent:
            sib = e+1
          # print(f'element {e} has sibling {sib}')
          #check whether sibling is active. If not action should be zero.

          if sib in active_grid:
             print(f'sibling {sib} is active')
          else:
            # print(f'sibling {sib} is NOT active')
            marks[index]=0
            ind += 1
            continue
          if(e < sib):
              sib_index = ind + 1
          elif(e > sib):
              sib_index = ind - 1




          # sib_index = np.where(active_grid == sib)[0][0]
          sib_nodes = intma[:,sib_index]
          sib_sols = q[sib_nodes]

          #Check whether sibling also meets derefinement criteria. If so, derefine both. If not, derefine neither.
          # if(max(sib_sols) >= 0.5):
          #   # print(f'sibling {sib} should not be derefined.')
          #   # marks[index]=0
          #   # marks[sib_index]=0
          if(max(sib_sols) < 0.5):
              # print(f'sibling {sib} should be derefined.')
              if sib not in defs:
                  marks[index]= -1
                  marks[sib_index]=-1
                  defs.append(e)
                  defs.append(sib)
              else:
                print(f'element {e} already marked for derefinement during sibling\'s marking.')

        ind+=1

        # else:
          # print(f'element {e} is NOT MARKED')




    return refs, defs, marks
    # print(f'marked')


# def refine(cur_grid, active, label_mat, info_mat, refs):
def refine(cur_grid, active, label_mat, info_mat, refs, marks):
    # cur_grid is is the pre-refined grid of coordinates at time of function call
    # active is the active cells at time of function call
    # refs is an array of elements marked for refinement. This function will refine those elements.


    # print(f'prerefined grid: {cur_grid}\n')

    ctr = 0
    ctr_max = len(refs)
    # print(f'ctr_max = {len(refs)}')
    # for elem in refs:
    if any(mark > 0 for mark in marks):
    # if any(marks):
        # for elem,mark in zip(active, marks):
        for i in range(len(active)):
            mark = marks[ctr]
            elem = active[ctr]
            if mark == 0 or mark == -1:
                ctr +=1
                continue
            #if mark != 0, then the following lines will be executed

            # print(f'ctr: {ctr}')
            # print(f'elem: {elem}')
            # print(f'mark: {mark}')

            # print(f'refine element: {elem}')
            parent = label_mat[elem-1][1]
            c1 = label_mat[elem-1][2]
            c2 = label_mat[elem-1][3]
            c1_l, c1_r = info_mat[c1-1][2],info_mat[c1-1][3]
            c2_l, c2_r = info_mat[c2-1][2],info_mat[c2-1][3]

            ref_index = np.where(active == elem)[0][0] + 1
            # print(f'element {elem} is at {ref_index-1}. That is, index {ref_index-1} of active is {active[ref_index-1]}')
            ref = np.insert(cur_grid, ref_index, c1_r)
            cur_grid = ref
            # Replace the value 3 with 30 and 31
            index = np.where(active == elem)[0][0]
            # print(f'here 4')
            # print(f'index: {index}')
            # print(f'active grid before refining element {elem} is {active}')
            active = np.delete(active,index)
            # print(f'here 5')
            # print(f'marks before refining element {elem} is {marks}')
            active = np.insert(active, index, [c1, c2])
            # print(f'new active grid after refining element {elem} is : {active}')
            marks = np.delete(marks,index)
            marks = np.insert(marks, index, [0, 0])
            # print(f'new marks after refining element {elem} is : {marks}')
            new_nelem = len(active)
            #     print(type(ref))
            new_npoin_cg = nop*new_nelem + 1
            new_npoin_dg = ngl*new_nelem
            ctr+=2
            # print(f'\n element number {elem} has parent {parent} and children {c1} and {c2}')

    else:
        # print(f'no elements to refine')
        ref = cur_grid
        new_nelem = len(active)
        new_npoin_cg = nop*new_nelem + 1
        new_npoin_dg = ngl*new_nelem
    # if(ctr == ctr_max):
    #   # print(f'no elements refined')
    #   ref = cur_grid
    #   new_nelem = len(active)
    # #     print(type(ref))
    #   new_npoin_cg = nop*new_nelem + 1
    #   new_npoin_dg = ngl*new_nelem


    return ref, active, marks, new_nelem, new_npoin_cg, new_npoin_dg

def derefine(cur_grid, active, label_mat, info_mat, defs, marks):
    # cur_grid is is the pre-refined grid of coordinates at time of function call
    # active is the active cells at time of function call
    # defs is an array of elements marked for refinement. This function will refine those elements.


    # ~~~~~~~~~~~> defs should already be paired up as siblings

    ctr = 0
    ctr_max = len(active)
    # print(f'ctr_max = {len(refs)}')
    # for elem in refs:
    if any(mark < 0 for mark in marks):
    # if any(marks):
        # for elem,mark in zip(active, marks):
        for i in range(ctr_max-1):
            mark = marks[ctr]
            elem = active[ctr]
            if mark == 0 or mark == 1:
                # print(f'element {elem} is not marked for refinement')
                ctr +=1
                continue

            #if mark != 0, then the following lines will be executed
            else:
                # print(f'ctr: {ctr}')
                # print(f'elem: {elem}')
                # print(f'mark: {mark}')

                # print(f'derefine element: {elem}')
                parent = label_mat[elem-1][1]
                c1 = label_mat[elem-1][2]
                c2 = label_mat[elem-1][3]
                c1_l, c1_r = info_mat[c1-1][2],info_mat[c1-1][3]
                c2_l, c2_r = info_mat[c2-1][2],info_mat[c2-1][3]

                def_index = np.where(active == elem)[0][0] + 1
                # print(f'element {elem} is at {def_index-1}. That is, index {def_index-1} of active is {active[def_index-1]}')
                if label_mat[elem-2][1] == parent:
                    sib = elem-1
                elif label_mat[elem][1] == parent:
                    sib = elem+1
                # print(f'element {elem} has sibling {sib}')
                # print(f'element {elem} has sibling {active[ctr+1]}')

                # cur_grid.pop(def_index + 1)
                # print(f'dleting grid element: {cur_grid[def_index]}')
                deref = np.delete(cur_grid, def_index)
                cur_grid = deref
                # cur_grid = ref
                # Replace the value 3 with 30 and 31
                index = np.where(active == elem)[0][0]
                # print(f'here 4')
                # print(f'index: {index}')
                #  print(f'active grid before derefining element {elem} and {sib} is {active}')
                active = np.delete(active,index)
                active = np.delete(active,index)
                # print(f'here 5')
                # print(f'marks before derefining element {elem} and {sib} is {marks}')
                active = np.insert(active, index, parent)
                # print(f'new active grid after refining element {elem} is : {active}')
                marks[index] = 0
                marks =  np.delete(marks,index+1)
                ctr_max -=1
                # print(f'new marks after refining element {elem} is : {marks}')
                new_nelem = len(active)
                #     print(type(ref))
                new_npoin_cg = nop*new_nelem + 1
                new_npoin_dg = ngl*new_nelem
                deref = cur_grid
                ctr+=1
            # print(f'\n element number {elem} has parent {parent} and children {c1} and {c2}')

    else:
        # print(f'no elements to derefine')
        deref = cur_grid
        new_nelem = len(active)
        new_npoin_cg = nop*new_nelem + 1
        new_npoin_dg = ngl*new_nelem


    return deref, active, marks, new_nelem, new_npoin_cg, new_npoin_dg