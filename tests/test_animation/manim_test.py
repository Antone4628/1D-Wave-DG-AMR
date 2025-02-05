# from manim import *
from manim import Scene, Axes, VGroup, VMobject, Dot, Text, Create, Transform
import numpy as np
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), 
    '..',
    '..'
))
sys.path.append(PROJECT_ROOT)

# Add the path to the folder containing ffmpeg to the PATH environment variable
os.environ['PATH'] += os.pathsep + '/Users/antonechacartegui/opt/anaconda3/envs/RL1/bin'


from numerical.solvers.dg_wave_solver import DGWaveSolver

class WaveEquationAMR(Scene):
    def construct(self):
        # Initialize solver
        xelem = np.array([-1, -0.4, 0, 0.4, 1])
        solver = DGWaveSolver(
            nop=4,
            xelem=xelem,
            max_elements=40,
            max_level=4,
            courant_max=0.1,
            icase=1
        )
        
        # Solve and collect results
        times, solutions, grids, coords = solver.solve(0.2)
        
        # Create axes
        axes = Axes(
            x_range=[-1.2, 1.2, 0.4],
            y_range=[-0.2, 1.4, 0.2],
            axis_config={"include_tip": False},
            x_axis_config={"numbers_to_include": np.arange(-1, 1.5, 0.5)},
            y_axis_config={"numbers_to_include": np.arange(0, 1.5, 0.5)}
        ).scale(0.8)
        
        # Add labels
        labels = axes.get_axis_labels(x_label="x", y_label="u(x,t)")
        
        # Create grid points as dots
        def create_grid_dots(grid):
            return VGroup(*[Dot(axes.c2p(x, 0), radius=0.03, color=BLUE) 
                          for x in grid])
            
        # Create solution curve
        def get_solution_points(coords, sol):
            return [axes.c2p(x, y) for x, y in zip(coords, sol)]
            
        # Initial setup
        grid_dots = create_grid_dots(grids[0])
        solution = VMobject(color=PURPLE)
        solution.set_points_smoothly(get_solution_points(coords[0], solutions[0]))
        
        # Time text
        time_text = Text("t = 0.000").scale(0.8)
        time_text.to_corner(UR)
        
        # Display initial state
        self.play(Create(axes), Create(labels))
        self.play(Create(grid_dots), Create(solution))
        self.play(Write(time_text))
        self.wait(0.5)
        
        # Animate solution evolution
        dt = times[1] - times[0]
        for i in range(1, len(times)):
            new_dots = create_grid_dots(grids[i])
            new_solution = VMobject(color="PURPLE")
            new_solution.set_points_smoothly(
                get_solution_points(coords[i], solutions[i])
            )
            new_time = Text(f"t = {times[i]:.3f}").scale(0.8)
            new_time.to_corner(UR)
            
            self.play(
                Transform(solution, new_solution),
                Transform(grid_dots, new_dots),
                Transform(time_text, new_time),
                run_time=dt
            )
            
        self.wait()

if __name__ == "__main__":
    scene = WaveEquationAMR()
    scene.render()