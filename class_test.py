"""
1D Wave Equation Solver with AMR

This script solves the 1D wave equation using a DG method with adaptive mesh refinement.
Uses the DGWaveSolver class for the core numerical solution.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numerical.solvers.dg_wave_solver import DGWaveSolver
# from numerical.solvers.dg_wave_solver import DGWaveSolver

# Define initial mesh
xelem = np.array([-1, -0.4, 0, 0.4, 1])
nelem = len(xelem) - 1

# Print initial mesh information
differences = np.diff(xelem)
print(f'Initial element sizes: {differences}')
print(f'Smallest initial element has dx: {np.min(differences)}')

# Solver parameters
max_level = 4         # Max level of refinement
nop = 4              # Polynomial order
courant_max = 0.1    # CFL number
time_final = 0.2    # Final time
icase = 1            # Test case number (1: Gaussian)

# Calculate smallest possible element size after refinement
dx_min = np.min(differences)/(2**max_level)
print(f'Smallest possible refined element: {dx_min}')

# Initialize solver
solver = DGWaveSolver(
    nop=nop,
    xelem=xelem,
    max_elements=40,
    max_level=max_level,
    courant_max=courant_max,
    icase=icase
)

print(f'Courant: {solver.wave_speed*solver.dt/dx_min:.6f}')
print(f'dt = {solver.dt:.6f}')
print(f'time_final = {time_final}')
print(f'timesteps = {time_final/solver.dt:.1f}')

# Solve and collect results
times, solutions, grids, coords = solver.solve(time_final)

# Create animation
plt.rcParams['animation.html'] = 'jshtml'
plt.style.use('ggplot')

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim([-1, 1])
ax.set_ylim([-0.1, 1.2])
ax.set_xticks(xelem)
ax.tick_params(axis='x', rotation=90, labelsize=8)
ax.set_title(f'{nelem} initial elements, full AMR to level {max_level}, dt = {solver.dt:.6f}')

# Add text annotations
frame_text = ax.text(0.05, 0.95, '',
                    horizontalalignment='left',
                    verticalalignment='top',
                    transform=ax.transAxes)
time_text = ax.text(0.05, 0.90, '',
                   horizontalalignment='left',
                   verticalalignment='top',
                   transform=ax.transAxes)

# Initialize plot
animated_plot = ax.plot(coords[0], solutions[0], color='darkmagenta')[0]

def update_data(frame):
    """Update function for animation."""
    animated_plot.set_ydata(solutions[frame])
    animated_plot.set_xdata(coords[frame])
    ax.set_xticks(grids[frame])
    frame_text.set_text('frame: %.1d' % frame)
    time_text.set_text('time: %.2f' % times[frame])

# Create animation
anim = FuncAnimation(
    fig=fig,
    func=update_data,
    frames=len(solutions),
    interval=10
)

# Save animation
gif_title = 'Solver_Class_1D_Wave_AMR_refdef_GIF.gif'
anim.save(gif_title, writer="pillow", fps=50)

plt.show()