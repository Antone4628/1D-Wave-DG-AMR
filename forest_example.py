import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches

from numerical.dg.basis import lgl_gen, Lagrange_basis
from numerical.dg.matrices import create_mass_matrix
from numerical.grid.mesh import create_grid_us
from numerical.amr.forest import *

xelem=np.array([-1, -0.4 ,0 ,0.4 ,1])

max_level = 3

label_mat, info_mat, active = forest(xelem, max_level)
print(f'label_mat {label_mat}')
print(f'info_mat {info_mat}')
print(f'active {active}')
