import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches

from numerical.dg.basis import lgl_gen, Lagrange_basis
from numerical.dg.matrices import *
from numerical.grid.mesh import create_grid_us
from numerical.amr.forest import forest
from numerical.solvers.utils import exact_solution
from numerical.solvers.wave import *

# from numerical.amr.adapt import adapt_mesh
# from numerical.amr.projection import create_S_matrix, create_scatters, create_gathers


# xelem=np.array([-1,  0 ,0.3 ,1])
# nelem = 3                 #Initial number of elements in level zero

# xelem=np.array([-1, -0.3 ,0 ,0.3 ,1])
xelem=np.array([-1, -0.4, 0 ,0.4 ,1])
nelem = 4                 #Initial number of elements in level zero


# xelem=np.array([-1, -0.6 ,-0.2, 0.2 ,0.6 ,1])
# nelem = 5                 #Initial number of elements in level zero

# xelem=np.array([-1, -0.9, -0.8, -0.7, -0.6, -0.5,  -0.4, -0.3, -0.2, -0.1
#                  ,0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 ,1])
# nelem = 20                 #Initial number of elements in level zero

# xelem=np.array([-1, -0.95, -0.9, -0.85, -0.8, -0.75, -0.7,-0.65, -0.6,-0.55, -0.5,
#                 -0.45,  -0.4, -0.35, -0.3, -0.25,-0.2, -0.15,-0.1,
#                 -0.05, 0,0.05, 0.1,0.15, 0.2,0.25, 0.3,0.35, 0.4,0.45, 0.5,0.55,
#                   0.6,0.65, 0.7,0.75, 0.8,0.85, 0.9 ,0.95,1])
# nelem = 40                 #Initial number of elements in level zero

differences = np.diff(xelem)
print(f'element sizes: {differences}')

# Find the minimum difference
min_interval = np.min(differences)
print(f'smallest element has dx: {min_interval}')

max_level = 4         #Max level of refinement
criterion = 1        #AMR Criterion type
cur_level = 0
nop = 4
ngl = nop + 1

dx_min = min_interval/(2**max_level)
print(f'smallest refined element: {dx_min}')
# dt_opt = Courant_max*dx_min/u

npoin_cg = nop*nelem + 1
npoin_dg = ngl*nelem



integration_points = 1      #=1 for LGL and =2 for LG
integration_type = 2        #=1 is inexact and =2 is exact
space_method_type = 'dg'    #CG or DG
flux_type = 2               #1=centered flux and 2=upwind

Courant_max = 0.1           #dt controlled by courant_max
time_final = .25        #final time in revolutions
iplot_solution = 1          #Switch to Plor of Not
iplot_matrices = 0          #??????

icase = 1                 #case number: 1 is a Gaussian, 2 is a square wave, 3 is a Gaussian with source, and 4 is a square wave with source
xmu = 0.05                  #filtering strength: 1 is full strength and 0 is no filter
ifilter = 0                 #time-step frequency that the filter is applied. 0=never, 1 = every time-step



#Compute Interpolation and Integration Points
xgl,wgl = lgl_gen(ngl)
if (integration_points ==1):
    integration_text = 'LGL'
    if (integration_type ==1):
        noq = nop
    elif (integration_type ==2):
        noq = nop + 1
    nq = noq + 1
    xnq,wnq = lgl_gen(nq)


psi, dpsi = Lagrange_basis(ngl,nq, xgl, xnq)
row_sum_psi = sum(psi)
row_sum_dpsi = sum(dpsi)

#Create Grid
coord_dg,  intma_dg, periodicity_dg  = create_grid_us(ngl,nelem,npoin_cg,npoin_dg,xgl, xelem)

#Form Global Matrix and Periodic BC Pointers
if (space_method_type == 'dg'):
    npoin = npoin_dg
    coord = coord_dg
    intma = intma_dg
    periodicity = periodicity_dg
    print(f'{space_method_type} : {integration_text}')

#Compute Exact Solution:
time = 0
qe, u = exact_solution(coord, npoin, time, icase)

#Compute Courant Number
dx = coord[1]-coord[0]

# dt = Courant_max*dx/u
dt_opt = Courant_max*dx_min/u
print(f'dt formula value: {dt_opt}')

# dt = dt/(2**max_level)
dt = dt_opt
Courant = u*dt/dx
Courant_min = u*dt/dx_min
print(f'Courant: {Courant_min:.6f}, dt = {dt:.6f}, time_final = {time_final}, timesteps = {time_final/dt:.1f}')

#Initialize State Vector:
q1=qe
q0=qe
qp=qe

#Get AMR data structures
label_mat, info_mat, active_grid = forest(xelem, max_level)


#Create Local/Element Mass and Differentiation Matrices
Me = create_mass_matrix(intma, coord, nelem, ngl, nq, wnq, psi)
De = create_diff_matrix(ngl, nq, wnq, psi, dpsi)

#Form Global Matrices
Mmatrix, Dmatrix = Matrix_DSS(Me, De, u, intma, periodicity, ngl, nelem, npoin)

#Apply BCs
if(flux_type == 2):
    Fmatrix = Fmatrix_upwind_flux(intma, nelem, npoin, ngl, u)

Rmatrix = Dmatrix - Fmatrix

#Left-Multiply by Inverse Mass Matrix
# Dmatrix_hat=Mmatrix\Rmatrix
Dmatrix_hat = np.linalg.solve(Mmatrix,Rmatrix)


#Time Integration
q0, time, plots, exact, grids, xelems = ti_LSRK_amr(q0, Dmatrix_hat, periodicity, xgl, xelem, wnq, xnq, psi, dpsi,u, time, time_final, dt, 1, max_level, criterion)
   
# coord, q0, qe, L2_norm, err, plots, grids, xelems = wave_solve(nop, nelem, xelem, integration_points, integration_type, space_method_type, icase, Courant_max, flux_type, time_final, max_level)


# Creatte animation
plt.rcParams['animation.html'] = 'jshtml'
plt.style.use('ggplot')
# grids = coord
# solutions = plots
solutions = plots
print(f'frames: {len(solutions)}')


fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim([-1,1])
ax.set_ylim([-0.1,1.2])
ax.set_xticks(xelem)
ax.tick_params(axis='x', rotation=90, labelsize=8)
ax.set_title(f'{nelem} initial elements, full AMR to level {max_level}, dt = {dt:.6f}')
frame_text = ax.text(0.05, 0.95,'',horizontalalignment='left',verticalalignment='top', transform=ax.transAxes)
time_text = ax.text(0.05, 0.90,'',horizontalalignment='left',verticalalignment='top', transform=ax.transAxes)


animated_plot = ax.plot(grids[0][:],solutions[0][:], color = 'darkmagenta')[0]

def update_data(frame):
    animated_plot.set_ydata(solutions[frame][:])
    animated_plot.set_xdata(grids[frame][:])
    ax.set_xticks(xelems[frame][:])
    frame_text.set_text('frame: %.1d' % frame)
    time = frame*dt
    time_text.set_text('time: %.2f' % time)

#     return animated_plot

anim = FuncAnimation(fig = fig,
                          func = update_data,
                          frames = len(solutions),
                          interval = 10)

gif_title = '1D_Wave_AMR_refdef'+'_GIF.gif'
# Save as gif file
anim.save(gif_title, writer = "pillow", fps=50 )
# HTML(anim.to_html5_video())
# content/drive/MyDrive/ColabNotebooks/Galerkin/

# plt.draw()
# plt.show()
# plt.draw()
anim
