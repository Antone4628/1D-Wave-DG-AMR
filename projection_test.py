import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches

from numerical.dg.basis import lgl_gen, Lagrange_basis
from numerical.dg.matrices import create_mass_matrix
from numerical.grid.mesh import create_grid_us
from numerical.amr.forest import forest
from numerical.amr.adapt import adapt_mesh, adapt_sol
from numerical.amr.projection import create_S_matrix, create_scatters, create_gathers, projections
from numerical.solvers.utils import exact_solution

# examine mass matrix
print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n New Test')
xelem0=np.array([-1, -0.4 ,0 ,0.4 ,1])
# xelem=np.array([-1.0 ,1.0])

integration_points = 1      #=1 for LGL and =2 for LG
integration_type = 2        #=1 is inexact and =2 is exact
space_method_type = 'dg'    #CG or DG
flux_type = 2

# nelem = 4
nelem0 = len(xelem0) - 1                 #Initial number of elements in level zero

nop = 4
ngl = nop + 1

icase = 1
max_level = 3

npoin_cg = nop*nelem0 + 1
npoin_dg0 = ngl*nelem0


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

# elif (integration_points == 2):
# print(f'\n ngl:{ngl}, nop: {nop}, nq: {nq}, len_xgl:{len(xgl)}')
psi, dpsi = Lagrange_basis(ngl,nq, xgl, xnq)

coord0,  intma0, periodicity0  = create_grid_us(ngl,nelem0,npoin_cg,npoin_dg0,xgl, xelem0)
print(f'nelem0: {nelem0}')
print(f'xelem0: {xelem0}')
print(f'npoin0: {npoin_dg0}')
print(f'coord0 length: {len(coord0)}')
print(f'intma0: {intma0}')


# print(f'intma:')

# print(f'coord: {coord}')
# print(f'psi:')
# display(psi)
# print(f'nelem: {nelem}, \nngl: {ngl}, \nnq: {nq}, \nwnq: {wnq}')

# emass0 = create_mass_matrix(intma0, coord0, nelem0, ngl, nq, wnq, psi)
# print(f'e_mass: {emass}')

ydots = np.zeros(len(coord0)) - 0.05

#  plt.scatter(coord, ydots)
q0, u = exact_solution(coord0, npoin_dg0, 0, icase)
print(f'\n q0: {q0}\n')

plt.ion()

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim([-1.2,1.2])
ax.set_ylim([-0.4,2.2])
ax.set_xticks(xelem0)
ax.scatter(coord0, ydots, color = 'darkcyan', label = 'unrefined nodes')
# ax.plot(coord,wave1, linewidth=10, color = 'lightblue', label = 'parent solution')
ax.axhline(y=0, color = 'darkgray')
for i in xelem0:
    ax.axvline(i, color = 'darkcyan')


label_mat, info_mat, active0 = forest(xelem0, max_level)
print(f'info_mat:\n {info_mat}')
print(f'active: {active0}')


marks0 = np.array([0,1,1,0])
refs = [2,3]
defs =[]
og_marks = marks0
og_active = active0
print(f'~~~~~~~~~~~~~~\n\n adapting mesh for first round')
xelem1, active1, new_marks, nelem1, new_npoin_cg, npoin_dg1 = adapt_mesh(nop, xelem0, active0, label_mat, info_mat, marks0)
coord1,  intma1, periodicity  = create_grid_us(ngl, nelem1, new_npoin_cg, npoin_dg1,xgl, xelem1)
# emass = create_mass_matrix(new_intma, new_coord, new_nelem, ngl, nq, wnq, psi)


y1 = np.zeros(len(coord1)) - 0.15
ax.scatter(coord1,y1, color = 'turquoise', label = 'first refinement')
for i in xelem1:
    ax.axvline(i, ymin=-0.1, ymax=0.9,color = 'turquoise', ls ='--',)






print(f'project data onto children:')

print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`\n\n')
print(f'creating projections using emass0, inttma0, coord0, nelem0')
# emass = create_mass_matrix(new_intma, new_coord, new_nelem, ngl, nq, wnq, psi)

emass0 = create_mass_matrix(intma0, coord0, nelem0, ngl, nq, wnq, psi)
print(f'emass0 shape: {np.shape(emass0)}')
PS1_0, PS2_0, PG1_0, PG2_0 = projections(emass0, intma0, coord0, nelem0, ngl, nq, wnq, xgl, xnq)

q1 = adapt_sol(q0, coord0, og_marks, og_active, label_mat, PS1_0, PS2_0, PG1_0, PG2_0, ngl)
print(f'done with new solution!')
print(f'new solution length: {len(q1)}')
print(f'new coord length: {len(coord1)}')

print(f'active1: {active1}')
print(f'coord1: {coord1}')
print(f'xelem1: {xelem1}')


# active = new_active
# xelem1 = cur_grid
print(f'nelem1: {nelem1}')
print(f'xelem1: {xelem1}')
print(f'npoin1: {npoin_dg1}')
print(f'coord1 length: {len(coord1)}')
print(f'intma1: {intma1}')

# print(f'new active: {active}')
# print(f'new marks: {new_marks}')

marks1 = ([0,1,1,0,0,0])
og_marks = marks1
og_active = active1
# emass = create_mass_matrix(new_intma, new_coord, new_nelem, ngl, nq, wnq, psi)

print(f'~~~~~~~~~~~~~~\n\n adapting mesh for second round')
xelem2, active2, new_marks, nelem2, new_npoin_cg, npoin_dg2 = adapt_mesh(nop, xelem1, active1, label_mat, info_mat, marks1)
coord2,  intma2, periodicity  = create_grid_us(ngl, nelem2, new_npoin_cg, npoin_dg2,xgl, xelem2)

# for i in xelem1:
#     ax.axvline(i, color = 'lightgray', ls ='--',)

# xelem2 = cur_grid
print(f'nelem2: {nelem2}')
print(f'xelem2: {xelem2}')
print(f'npoin2: {npoin_dg2}')
print(f'coord2 length: {len(coord2)}')
print(f'intma2: {intma2}')

print(f'active2: {active2}')
# print(f'new marks: {new_marks}')


print(f'creating emass for second time to use on second projection. using intma1: {intma1}, \ncoord1:{coord1}, \nnelem1: {nelem1}')

# emass1 = create_mass_matrix(intma2, coord2, nelem2, ngl, nq, wnq, psi)
emass1 = create_mass_matrix(intma1, coord1, nelem1, ngl, nq, wnq, psi)

print(f'emass1 shape: {np.shape(emass1)}')

print(f'project data onto children:')

print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`\n\n')
print(f'creating projections using emass1, intma1, coord1, nelem1')
# PS1_1, PS2_1, PG1_1, PG2_1 = projections(emass1, intma2, coord2, nelem2, ngl, nq, wnq, xgl, xnq)
PS1_1, PS2_1, PG1_1, PG2_1 = projections(emass1, intma1, coord1, nelem1, ngl, nq, wnq, xgl, xnq)


q2 = adapt_sol(q1, coord1, og_marks, og_active, label_mat, PS1_1, PS2_1, PG1_1, PG2_1, ngl)

print(f'q_ad: {np.shape(q2)}')
print(f'new_cord: {np.shape(coord2)}')

y2 = np.zeros(len(coord2))-0.25
ax.scatter(coord2,y2, color = 'darkmagenta', label = 'second refinement')
for i in xelem2:
    ax.axvline(i, ymin=-0.2, ymax=0.8,color = 'darkmagenta', ls ='--',)


# active = active2

# ax.plot(coord0,q0, linewidth=10, color = 'lightblue', label = 'parent solution')

# ax.plot(coord2, q2, color = 'blue',marker = 11,  ls = '--', label = 'scattered')
# ax.legend(loc="upper right")

# plt.show()


#now gather q7 and q8 back to parent q2:

#coarsent the current two-element mesh
# PG1, PG2 = create_gathers(emass, S1, S2)
# PG1 = np.reshape(PG1,(ngl,ngl))
# PG2 = np.reshape(PG2,(ngl,ngl))

# marks2 = np.array([0,-1,-1,-1,-1,0,0,0])
# marks2 = np.array([0,-1,-1,1,1,0,0,0])
marks2 = np.array([0,-1,-1,1,1,-1,-1,0])

og_marks = marks2
og_active = active2
xelem3, active3, new_marks, nelem3, new_npoin_cg, npoin_dg3 = adapt_mesh(nop, xelem2, active2, label_mat, info_mat, marks2)
coord3,  intma3, periodicity  = create_grid_us(ngl, nelem3, new_npoin_cg, npoin_dg3,xgl, xelem3)

print(f'nelem3: {nelem3}')
print(f'xelem3: {xelem3}')
print(f'npoin3: {npoin_dg3}')
print(f'coord3 length: {len(coord3)}')
print(f'intma3: {intma3}')

print(f'active3: {active3}')
# print(f'new marks: {new_marks}')


# print(f'creating emass2 for third time to use on third projection (gather). using intma3: {intma3}, \ncoord3:{coord3}, \nnelem3: {nelem3}')
# emass2 = create_mass_matrix(intma3, coord3, nelem3, ngl, nq, wnq, psi)
print(f'creating emass2 for third time to use on third projection (gather). using intma2: {intma2}, \ncoord2:{coord2}, \nnelem2: {nelem2}')
emass2 = create_mass_matrix(intma2, coord2, nelem2, ngl, nq, wnq, psi)
print(f'emass12shape: {np.shape(emass2)}')
# PS1_2, PS2_2, PG1_2, PG2_2 = projections(emass2, intma3, coord3, nelem3, ngl, nq, wnq, xgl, xnq)
PS1_2, PS2_2, PG1_2, PG2_2 = projections(emass2, intma2, coord2, nelem2, ngl, nq, wnq, xgl, xnq)



q_gath = adapt_sol(q2, coord2, og_marks, og_active, label_mat, PS1_2, PS2_2, PG1_2, PG2_2, ngl)

y3 = np.zeros(len(coord3))-0.35
ax.scatter(coord3,y3, color = 'magenta', label = 'third refinement')
for i in xelem3:
    ax.axvline(i, ymin=-0.3, ymax=0.7, color = 'magenta', ls ='--',)

ax.plot(coord0,q0, linewidth=10, color = 'darkcyan', label = 'unrefined solution')
ax.plot(coord1, q1,linewidth=8, color = 'turquoise', label = 'refined 1')

# ax.plot(coord2, q2, color = 'darkmagenta',marker = 11,  ls = '--', label = 'refined 2')
ax.plot(coord2, q2, linewidth=4, color = 'darkmagenta', label = 'refined 2')
ax.legend(loc="upper right")

# active = new_active

# print(f'PG1: {np.shape(PG1)}')
# print(f'PG2: {np.shape(PG2)}')
# print(f'q1: {np.shape(q7)}')
# print(f'q2: {np.shape(q8)}')

# q2l = np.dot(PG1[2],q7) 
# q2r = np.dot(PG2[2],q8)

# print(f'ql: {np.shape(q2l)}')
# print(f'qr: {np.shape(q2r)}')

# q2g = q2l + q2r

# print(f'gathered q: {q2g}')

# q3l = np.dot(PG1[3],q9) 
# q3r = np.dot(PG2[3],q10)

# print(f'ql: {np.shape(q3l)}')
# print(f'qr: {np.shape(q3r)}')

# q3g = q3l + q3r

# print(f'gathered q: {q3g}')

# pars = np.concatenate((q2g,q3g))
# wave_gathered = np.concatenate((wave1[:ngl], pars, wave1[3*ngl:]))
# print(f'wave_gathered length: {len(wave_gathered)}')

# ax.plot(coord, wave_gathered, color = 'red', label = 'gathered')
ax.plot(coord3, q_gath, color = 'magenta', ls = '--', label = 'refined 3')
ax.legend(loc="upper right")


# marks = np.array([-1,-1])
# refs = []
# defs =[2,3]

# cur_grid, active, marks, new_nelem, new_npoin_cg, new_npoin_dg = adapt_mesh(nop, xelem, active, label_mat, info_mat, marks)
# new_coord,  new_intma, periodicity  = create_grid_us(ngl, new_nelem, new_npoin_cg, new_npoin_dg,xgl, cur_grid)

# # active = [1,2,3,4]
# refined, active, marks, new_nelem, new_npoin_cg, new_npoin_dg = refine(nop,xelem, active, label_mat, info_mat, refs, marks)
# ref_coord,  new_intma, periodicity  = create_grid_us(ngl,new_nelem,new_npoin_cg,new_npoin_dg,xgl, refined)

# newdots = np.zeros(len(ref_coord)) + 0.1
# ax.scatter(ref_coord, newdots, color = 'red')
# for i in refined:
#     ax.axvline(i, color = 'lightgray')

# print(f'new intma:')




# wave2 = wave1
# print(intma[:,refs[0]-1])
# nodes0 = intma[:,refs[0]-1]

# S1, S2 = create_S_matrix(intma, coord, nelem, ngl, nq, wnq)
# PS1, PS2 = create_scatters(emass, S1, S2)
# print(f'PS1:')

# print(f'PS2:')
