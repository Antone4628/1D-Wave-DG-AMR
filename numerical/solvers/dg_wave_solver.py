"""
DG Wave Solver with Adaptive Mesh Refinement

This module implements a Discontinuous Galerkin solver for the 1D wave equation
with h-adaptation capabilities using hierarchical mesh refinement.
"""

import numpy as np
from ..dg.basis import lgl_gen, Lagrange_basis
from ..dg.matrices import (create_mass_matrix, create_diff_matrix, 
                      Fmatrix_upwind_flux, Matrix_DSS, create_RM_matrix)
from ..grid.mesh import create_grid_us
from ..amr.forest import forest, mark
from ..amr.adapt import adapt_mesh, adapt_sol
from ..amr.projection import projections
from .utils import exact_solution

class DGWaveSolver:
    """
    Discontinuous Galerkin solver for 1D wave equation with AMR capabilities.
    
    Attributes:
        nop (int): Polynomial order
        ngl (int): Number of LGL points (nop + 1)
        nelem (int): Number of elements
        xelem (array): Element boundary coordinates
        max_level (int): Maximum refinement level
        dt (float): Time step size
        time (float): Current simulation time
    """
    
    # def __init__(self, nop, xelem, max_level, courant_max=0.1, icase=1):
    def __init__(self, nop, xelem, max_elements, max_level, courant_max=0.1, icase=1):
        """
        Initialize the DG solver.
        
        Args:
            nop (int): Polynomial order
            xelem (array): Initial element boundary coordinates
            max_level (int): Maximum refinement level
            courant_max (float): Maximum Courant number
            icase (int): Test case number
        """
        # Basic parameters
        
        self.nop = nop
        self.xelem = xelem
        self.max_elements = max_elements
        self.ngl = nop + 1
        self.nelem = len(xelem) - 1
        self.max_level = max_level
        self.icase = icase
        self.time = 0.0

        # Calculate minimum dx based on max_level
        self.dx_min = np.min(np.diff(xelem)) / (2**max_level)
        
        # Compute points and weights
        self.xgl, self.wgl = lgl_gen(self.ngl)
        self.nq = self.nop + 2  # Exact integration
        self.xnq, self.wnq = lgl_gen(self.nq)
        
        # Initialize basis functions
        self.psi, self.dpsi = Lagrange_basis(self.ngl, self.nq, self.xgl, self.xnq)
        
        # Initialize grid and mesh structures
        self._initialize_mesh()
        
        # Initialize solution
        self.q = self._initialize_solution()
        
        # Compute optimal timestep
        self._compute_timestep(courant_max)
        
        # Initialize projection matrices for AMR
        self._initialize_projections()
        
    def _initialize_mesh(self):
        """Initialize mesh and grid structures."""
        # Compute basic mesh parameters
        self.npoin_cg = self.nop * self.nelem + 1
        self.npoin_dg = self.ngl * self.nelem
        
        # Create hierarchical mesh structure for AMR
        self.label_mat, self.info_mat, self.active = forest(self.xelem, self.max_level)
        
        # Create computational grid
        self.coord, self.intma, self.periodicity = create_grid_us(
            self.ngl, self.nelem, self.npoin_cg, self.npoin_dg, 
            self.xgl, self.xelem
        )
        
    def _initialize_solution(self):
        """Initialize solution vector."""
        # Get initial condition from exact solution
        q, self.wave_speed = exact_solution(
            self.coord, self.npoin_dg, self.time, self.icase
        )
        return q
        
    def _compute_timestep(self, courant_max):
        """Compute optimal timestep based on CFL condition."""
        dx_min = np.min(np.diff(self.xelem)) / (2**self.max_level)
        self.dt = courant_max * dx_min / self.wave_speed
        
    def _initialize_projections(self):
        """Initialize projection matrices for AMR."""
        # Get reference mass matrix
        RM = create_RM_matrix(self.ngl, self.nq, self.wnq, self.psi)
        
        # Create projection matrices
        self.PS1, self.PS2, self.PG1, self.PG2 = projections(
            RM, self.ngl, self.nq, self.wnq, self.xgl, self.xnq
        )
        
    def _update_matrices(self):
        """Update system matrices based on current mesh."""
        # Create element matrices
        self.Me = create_mass_matrix(
            self.intma, self.coord, self.nelem, self.ngl, 
            self.nq, self.wnq, self.psi
        )
        self.De = create_diff_matrix(self.ngl, self.nq, self.wnq, self.psi, self.dpsi)
        
        # Create global matrices
        self.M, self.D = Matrix_DSS(
            self.Me, self.De, self.wave_speed, self.intma, 
            self.periodicity, self.ngl, self.nelem, self.npoin_dg
        )
        
        # Create flux matrix and spatial operator
        self.F = Fmatrix_upwind_flux(
            self.intma, self.nelem, self.npoin_dg, self.ngl, self.wave_speed
        )
        R = self.D - self.F
        
        # Create spatial operator
        self.Dhat = np.linalg.solve(self.M, R)
        
    def adapt_mesh(self, criterion=1, marks_override = None):
        """
        Perform mesh adaptation based on solution criteria.
        
        Args:
            criterion (int): AMR marking criterion
        """
        # Get refinement marks
        marks = mark(self.active, self.label_mat, self.intma, self.q, criterion)
        
        # Store pre-adaptation state
        pre_grid = self.xelem
        pre_active = self.active
        pre_nelem = self.nelem
        pre_coord = self.coord
        pre_npoin_dg = self.npoin_dg
        
        # Adapt mesh
        new_grid, new_active, _, new_nelem, npoin_cg, new_npoin_dg = adapt_mesh(
            self.nop, pre_grid, pre_active, self.label_mat, 
            self.info_mat, marks
        )
        
        # Update grid
        new_coord, new_intma, new_periodicity = create_grid_us(
            self.ngl, new_nelem, npoin_cg, new_npoin_dg, 
            self.xgl, new_grid
        )
        
        # Project solution
        q_new = adapt_sol(
            self.q, pre_coord, marks, pre_active, self.label_mat,
            self.PS1, self.PS2, self.PG1, self.PG2, self.ngl
        )
        
        # Update solver state
        self.q = q_new
        self.active = new_active
        self.nelem = new_nelem
        self.intma = new_intma
        self.coord = new_coord
        self.xelem = new_grid
        self.npoin_dg = new_npoin_dg
        self.periodicity = new_periodicity
        
        # Update matrices for new mesh
        self._update_matrices()
        
    def step(self, dt=None):
        """
        Take one time step using low-storage RK method.
        
        Args:
            dt (float, optional): Time step size. Uses self.dt if None.
        """
        if dt is None:
            dt = self.dt
            
        # RK coefficients
        RKA = np.array([0,
                       -567301805773.0/1357537059087,
                       -2404267990393.0/2016746695238,
                       -3550918686646.0/2091501179385,
                       -1275806237668.0/842570457699])
        
        RKB = np.array([1432997174477.0/9575080441755,
                       5161836677717.0/13612068292357,
                       1720146321549.0/2090206949498,
                       3134564353537.0/4481467310338,
                       2277821191437.0/14882151754819])
        
        # Initialize stage values
        dq = np.zeros(self.npoin_dg)
        qp = self.q.copy()
        
        # RK stages
        for s in range(len(RKA)):
            # Compute RHS
            R = self.Dhat @ qp
            
            # Update stage values
            for i in range(self.npoin_dg):
                dq[i] = RKA[s]*dq[i] + dt*R[i]
                qp[i] = qp[i] + RKB[s]*dq[i]
                
            # Apply periodicity
            if self.periodicity[-1] == self.periodicity[0]:
                qp[-1] = qp[0]
                
        # Update solution and time
        self.q = qp
        self.time += dt
        
    def solve(self, time_final):
        """
        Solve to specified final time.
        
        Args:
            time_final (float): Final simulation time
            
        Returns:
            tuple: (times, solutions, grids, coords)
                times: Time points
                solutions: Solution at each time
                grids: Element boundaries at each time
                coords: Node coordinates at each time
        """
        times = [self.time]
        solutions = [self.q.copy()]
        grids = [self.xelem.copy()]
        coords = [self.coord.copy()]
        
        while self.time < time_final:
            # Adjust final step if needed
            dt = min(self.dt, time_final - self.time)
            
            # Adapt mesh if needed
            self.adapt_mesh()
            
            # Take time step
            self.step(dt)
            
            # Store results
            times.append(self.time)
            solutions.append(self.q.copy())
            grids.append(self.xelem.copy())
            coords.append(self.coord.copy())
            
        return times, solutions, grids, coords

    def get_exact_solution(self):
        """Get exact solution at current time."""
        qe, _ = exact_solution(self.coord, self.npoin_dg, self.time, self.icase)
        return qe
    
    def reset(self):
        """Reset solver to initial state."""
        # Reinitialize mesh
        self.nelem = len(self.xelem) - 1
        
        # Recalculate number of points
        self.npoin_cg = self.nop * self.nelem + 1
        self.npoin_dg = self.ngl * self.nelem
        
        self.label_mat, self.info_mat, self.active = forest(self.xelem, self.max_level)
        
        # Reset grid
        self.coord, self.intma, self.periodicity = create_grid_us(
            self.ngl, self.nelem, self.npoin_cg, self.npoin_dg, self.xgl, self.xelem
        )
        
        # Reset solution to initial condition
        self.q, _ = exact_solution(self.coord, self.npoin_dg, 0.0, self.icase)
        
        # Reset time
        self.time = 0.0
        
        # Update matrices for initial mesh
        self._update_matrices()
        
        return self.q
    # def reset(self):
    #     """Reset solver to initial state."""
    #     # Reinitialize mesh
    #     self.nelem = len(self.xelem) - 1
    #     self.label_mat, self.info_mat, self.active = forest(self.xelem, self.max_level)
        
    #     # Reset grid
    #     self.coord, self.intma, self.periodicity = create_grid_us(
    #         self.ngl, self.nelem, self.npoin_cg, self.npoin_dg, self.xgl, self.xelem
    #     )
        
    #     # Reset solution to initial condition
    #     self.q, _ = exact_solution(self.coord, self.npoin_dg, 0.0, self.icase)
    
    #     return self.q