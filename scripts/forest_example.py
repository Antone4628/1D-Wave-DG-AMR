import numpy as np
import os
import sys
import traceback

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), 
    # '..',
    '..'
))
sys.path.append(PROJECT_ROOT)


from numerical.amr.forest_documented import *

xelem=np.array([-1, -0.4 ,0 ,0.4 ,1])

max_level = 3

label_mat, info_mat, active = forest(xelem, max_level)
print(f'label_mat {label_mat}')
print(f'info_mat {info_mat}')
print(f'active {active}')
