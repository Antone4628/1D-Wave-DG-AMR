import numpy as np
import matplotlib.pyplot as plt
from time import sleep
from numpy.linalg import norm
from ..dg.matrices import create_mass_matrix, create_diff_matrix, Fmatrix_upwind_flux, Matrix_DSS, create_RM_matrix
from ..dg.basis import *
from ..grid.mesh import create_grid_us
<<<<<<< HEAD
from ..amr.forest import forest, mark, get_active_levels, print_active_levels
from ..amr.adapt import adapt_mesh, adapt_sol, enforce_2_1_balance, print_balance_violations
=======
from ..amr.forest import forest
from ..amr.adapt import adapt_mesh, adapt_sol,mark
>>>>>>> clean
from ..amr.projection import*
from .utils import *


def ti_LSRK_amr(q0, Dhat, periodicity, xgl, xelem, wnq, xnq, psi, dpsi,u, time, time_final, dt, 
                icase, max_level, criterion):
    """
    Low-storage Runge-Kutta time integration with AMR.
    
    Args:
        q0 (array): Initial solution
        Dhat (array): Spatial operator matrix
        periodicity (array): Periodic boundary mapping
        xgl (array): LGL nodes
        xelem (array): Element boundaries
        wnq (array): Quadrature weights
        psi, dpsi (array): Basis functions and derivatives
        u (float): Wave speed
        time (float): Current time
        time_final (float): End time
        dt (float): Time step
        icase (int): Test case
        max_level (int): Max refinement level
        criterion (int): Marking criterion
        
    Returns:
        tuple: (q0, time, plots, exact, grids, xelems)
    """
# qe, u = exact_solution(coord, npoin, time, icase)

    ngl = int(len(xgl))
    nop = ngl - 1
    nq  = int(len(wnq))
    nelem = int(len(q0)/ngl)
    print(f'nelem: {nelem}')
    npoin_cg = nop*nelem + 1
    npoin_dg = ngl*nelem
    Npoin = npoin_dg
    label_mat, info_mat, active = forest(xelem, max_level)
    coord,  intma, periodicity  = create_grid_us(ngl,nelem,npoin_cg, Npoin ,xgl, xelem)
    qe, u = exact_solution(coord, npoin_dg, time, icase)


#     print(f'timesteps: {time_final/dt}')
    frames = np.ceil(time_final/dt)
#     print(f'frames: {frames}')
    cols = int(len(q0))
    rows = int(frames)
    # plots = np.zeros((rows, cols))
    plots = []
    exact = []
    grids = []
    xelems = []
    # exact = np.zeros((rows, cols))




    RKA = np.array([0,
       (-567301805773) / (1357537059087),
       (-2404267990393) / (2016746695238),
       (-3550918686646) / (2091501179385),
       (-1275806237668) / (842570457699 )])

    RKB = np.array([(1432997174477) / (9575080441755 ),
       (5161836677717) / (13612068292357),
       (1720146321549) / (2090206949498 ),
       (3134564353537) / (4481467310338 ),
       (2277821191437) / (14882151754819)])

    RKC = np.array([0,
       (1432997174477) / (9575080441755),
       (2526269341429) / (6820363962896),
       (2006345519317) / (3224310063776),
       (2802321613138) / (2924317926251)])

    Npoin = len(q0) #also use for npoin_dg
#     print(f'Npoin = {Npoin}')
    # dq = np.zeros(Npoin)
    qp=q0
    stages=len(RKA)
#     print(f'stages = {stages}')

    #time integration:
    anim = 0
    grid = xelem


    # # Get Red Mass matrix and projection matrices
    RM = create_RM_matrix(ngl, nq, wnq, psi)
    PS1, PS2, PG1, PG2 = projections(RM, ngl, nq, wnq, xgl, xnq)

    # # first fram will be IC
    plots.append(qp.copy())
    exact.append(qe.copy())
    grids.append(coord.copy())
    xelems.append(grid.copy())

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # level = 0
    # while(level <= max_level):
    #     # Get refinement marks
    #     marks = mark(active, label_mat, intma, qp)

    #     pre_marks = marks
    #     pre_active = active  
    #     pre_grid = grid
    #     pre_nelem = nelem
    #     pre_intma = intma
    #     pre_coord = coord
    #     pre_npoin_dg = npoin_dg


    #     # Adapt mesh
    #     new_grid, new_active, ref_marks, new_nelem, npoin_cg, new_npoin_dg = adapt_mesh(nop, pre_grid, pre_active, label_mat, info_mat, marks)
    #     new_coord, new_intma, periodicity = create_grid_us(ngl, new_nelem, npoin_cg, new_npoin_dg, xgl, new_grid)
        

    #     # Project solution
    #     # q_ad = adapt_sol(qp, pre_coord, marks, pre_active, label_mat, PS1, PS2, PG1, PG2, ngl)

    #     # Update for next level
    #     # qp = q_ad
    #     active = new_active
    #     nelem = new_nelem
    #     intma = new_intma
    #     coord = new_coord
    #     grid = new_grid
    #     npoin_dg = new_npoin_dg

    #     qp, u = exact_solution(new_coord, new_npoin_dg, time, icase)



    #     level += 1
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # qp, u = exact_solution(new_coord, new_npoin_dg, time, icase)
    # plots.append(qp.copy())
    # exact.append(qe.copy())
    # grids.append(coord.copy())
    # xelems.append(grid.copy())



    # qp, u = exact_solution(coord, npoin_dg, time, icase)
    # # qp = qe


    while (time < time_final):
        time = time + dt
        if (time > time_final):
            time = time -dt
            dt = time_final-time
            time = time+dt

        #~~~~~~~~~~~~~~~~

        #Insert refinement routines here

        level = 0
        while(level <= max_level):
        #     # Get refinement marks
            marks = mark(active, label_mat, intma, qp, criterion)

            # print(f'pre ratio enforcement marks: {marks}')

<<<<<<< HEAD
            marks = enforce_2_1_balance(label_mat, active, marks)

 
=======
        
>>>>>>> clean

            # print(f'post ratio enforcement marks: {marks}')

            pre_marks = marks
            pre_active = active  
            pre_grid = grid
            pre_nelem = nelem
            pre_intma = intma
            pre_coord = coord
            pre_npoin_dg = npoin_dg


            # Adapt mesh
            # print(f'pre-adaptation:')
            # get_active_levels(active, label_mat)
            # print_active_levels(active, label_mat)


            new_grid, new_active, ref_marks, new_nelem, npoin_cg, new_npoin_dg = adapt_mesh(nop, pre_grid, pre_active, label_mat, info_mat, marks)
            new_coord, new_intma, periodicity = create_grid_us(ngl, new_nelem, npoin_cg, new_npoin_dg, xgl, new_grid)


<<<<<<< HEAD
            # Print state after enforcement
            print_balance_violations(new_active, label_mat)
            
            # print(f'post-adaptation:')
=======
>>>>>>> clean
            # get_active_levels(new_active, label_mat)
            # print_active_levels(new_active, label_mat)

            # Project solution
            q_ad = adapt_sol(qp, pre_coord, marks, pre_active, label_mat, PS1, PS2, PG1, PG2, ngl)

            # Update for next level
            qp = q_ad
            active = new_active
            nelem = new_nelem
            intma = new_intma
            coord = new_coord
            grid = new_grid
            npoin_dg = new_npoin_dg

            # qe, u = exact_solution(new_coord, new_npoin_dg, time, icase)

            # q0 = qp


            # plots.append(qp.copy())
            # exact.append(qe.copy())
            # grids.append(coord.copy())
            # xelems.append(grid.copy())
            
            level += 1
        

        Me = create_mass_matrix(intma, coord, nelem, ngl, nq, wnq, psi)
        # print(f'Me good')
        De = create_diff_matrix(ngl, nq, wnq, psi, dpsi)
        # print(f'De good')
        Mmatrix, Dmatrix = Matrix_DSS(Me, De, u, intma, periodicity, ngl, nelem, npoin_dg)
        # print(f'DSS good')
        Fmatrix = Fmatrix_upwind_flux(intma, nelem, npoin_dg, ngl, u)
        Rmatrix = Dmatrix - Fmatrix
        
        Dhat = np.linalg.solve(Mmatrix,Rmatrix)
        
        dq = np.zeros(npoin_dg)


        #~~~~~~~~~~~~~~~
        # plots.append(qp)
        # exact.append(qe)
        # grids.append(coord)
        # xelems.append(grid)

        #RK stages
        for s in range(stages):
            #Create RHS Matrix
            R = Dhat@qp #only valid for cg
            #solve system
            for I in range(npoin_dg):
#                 print(f'I: {I}')
#                 print(f'dt: {dt}')
                dq[I] = RKA[s]*dq[I] + dt*R[I]
                qp[I] = qp[I] + RKB[s]*dq[I]

            if(periodicity[-1] == periodicity[0]):
                qp[-1]=qp[0]



        # print(f'time step: {anim}')
        q0 = qp
        # qe, u = exact_solution(coord, npoin, time, icase)
        #         print(f'timestep: {anim}\n plot: {qp}')
        #         if (anim%10 == 0):
        # plots[anim][:] = q0
        # exact[anim][:] = qe



        plots.append(q0.copy())
        exact.append(qe.copy())
        grids.append(coord.copy())
        xelems.append(grid.copy())



        # print(f'\ntimestep: {anim}\n active: {active}\n marks: {marks}')
        #         print(f'plots[i]: {plots[anim]}')
        anim +=1





    return q0, time, plots, exact, grids, xelems, 






# def wave_solve(nop, nelem, xelem, integration_points, integration_type, space_method_type, icase, Courant_max, flux_type,time_final, max_level):
#     """
#     Main solver for wave equation with adaptive mesh refinement.
    
#     Args:
#         nop (int): Polynomial order
#         nelem (int): Initial number of elements
#         xelem (array): Element boundary coordinates
#         integration_points (int): 1=LGL, 2=LG
#         integration_type (int): 1=inexact, 2=exact
#         space_method_type (str): 'dg' or 'cg'
#         icase (int): Test case number
#         Courant_max (float): Maximum Courant number
#         flux_type (int): Type of numerical flux
#         time_final (float): Final simulation time
#         max_level (int): Maximum refinement level
        
#     Returns:
#         tuple: (coord, q0, qe, L2_norm, err, exact, grids, xelems)
#     """
#     ngl = nop + 1

#     npoin_cg = nop*nelem + 1
#     npoin_dg = ngl*nelem

#     #Compute Interpolation and Integration Points
#     xgl,wgl = lgl_gen(ngl)

#     if (integration_points ==1):
#         integration_text = 'LGL'
#         if (integration_type ==1):
#             noq = nop
#         elif (integration_type ==2):
#             noq = nop + 1
#         nq = noq + 1
#         xnq,wnq = lgl_gen(nq)

#     # elif (integration_points == 2):

#     psi, dpsi = Lagrange_basis(ngl,nq, xgl, xnq)
#     row_sum_psi = sum(psi)
#     row_sum_dpsi = sum(dpsi)
# #     print(f'row sum psi {row_sum_psi}')
# #     print(f'row sum dpsi {row_sum_dpsi}')

#     #Create Grid
# #     coord_cg, coord_dg, intma_cg,  intma_dg,  periodicity_cg, periodicity_dg  = create_grid(ngl,nelem,npoin_cg,npoin_dg,xgl)
# #     coord_cg, coord_dg, intma_cg,  intma_dg,  periodicity_cg, periodicity_dg  = create_grid_us(ngl,nelem,npoin_cg,npoin_dg,xgl, xelem)
#     coord_dg, intma_dg,  periodicity_dg  = create_grid_us(ngl,nelem,npoin_cg,npoin_dg,xgl, xelem)

#     #Form Global Matrix and Periodic BC Pointers
#     if (space_method_type == 'dg'):
#         npoin = npoin_dg
#         coord = coord_dg
#         intma = intma_dg
#         periodicity = periodicity_dg
# #         print(f'{space_method_type} : {integration_text}')

#     #Compute Exact Solution:
#     time = 0
#     qe, u = exact_solution(coord, npoin, time, icase)

#     #Compute Courant Number
#     dx = coord[1]-coord[0]
#     dt = Courant_max*dx/u
#     # dt = 0.002
#     Courant = u*dt/dx
# #     print(f'Courant = {Courant}, dt = {dt}, time_final = {time_final}')

#     #Create Local/Element Mass and Differentiation Matrices
#     Me = create_mass_matrix(intma, coord, nelem, ngl, nq, wnq, psi)
#     De = create_diff_matrix(ngl, nq, wnq, psi, dpsi)

#     #Form Global Matrices
#     Mmatrix, Dmatrix = Matrix_DSS(Me, De, u, intma, periodicity, ngl, nelem, npoin)

#     #Apply BCs
#     if(flux_type == 2):
#         Fmatrix = Fmatrix_upwind_flux(intma, nelem, npoin, ngl, u)

#     Rmatrix = Dmatrix - Fmatrix

#     #Left-Multiply by Inverse Mass Matrix
#     # Dmatrix_hat=Mmatrix\Rmatrix
#     Dmatrix_hat = np.linalg.solve(Mmatrix,Rmatrix)

#     #Initialize State Vector:
#     q1=qe
#     q0=qe
#     qp=qe

#     #Time Integration
#     # q0, time, plots = ti_LSRK(q0, Dmatrix_hat, periodicity, time, time_final, dt)
#     q0, time, plots, exact, grids, xelems = ti_LSRK_amr(q0, Dmatrix_hat, periodicity, xgl, xelem, wnq, xnq, psi, dpsi,u, time, time_final, dt, 1, max_level)
# #     q0, time, plots = ti_LSRK_amr(q0, Dmatrix_hat, periodicity_dg, xgl, xelem, wnq, psi, dpsi, u, time, time_final, dt)

#     #Compute Exact Solution:
#     # qe, u = exact_solution(coord, npoin, time, icase)


#     #Compute Norm
#     # L2_norm = norm(q0-qe)/norm(qe)
#     L2_norm = []
#     # err = L2_err_norm(nop, nelem, q0, qe)
#     err = []

#     return coord, q0, qe, L2_norm, err, plots, grids, xelems
