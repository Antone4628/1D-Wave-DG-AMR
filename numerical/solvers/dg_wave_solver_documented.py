"""
Discontinuous Galerkin Wave Solver with Adaptive Mesh Refinement

This module implements a high-order Discontinuous Galerkin (DG) solver for the 1D wave equation
with h-adaptation capabilities using hierarchical mesh refinement. The solver uses:
- Legendre-Gauss-Lobatto (LGL) nodal basis functions
- Upwind numerical fluxes for interface treatment
- Low-storage Runge-Kutta time integration
- Hierarchical mesh refinement with solution projection
"""

import numpy as np
from ..dg.basis import lgl_gen, Lagrange_basis
from ..dg.matrices import (create_mass_matrix, create_diff_matrix, 
                      Fmatrix_upwind_flux, Matrix_DSS, create_RM_matrix)
from ..grid.mesh import create_grid_us
<<<<<<< HEAD
from ..amr.forest import forest, mark
from ..amr.adapt import adapt_mesh, adapt_sol
=======
from ..amr.forest import forest
from ..amr.adapt import adapt_mesh, adapt_sol, mark
>>>>>>> clean
from ..amr.projection import projections
from .utils import exact_solution

class DGWaveSolver:
    """
    Discontinuous Galerkin solver for 1D wave equation with Adaptive Mesh Refinement (AMR).
    
    This solver implements:
    - Modal DG discretization with LGL nodes
    - Hierarchical h-refinement for mesh adaptation
    - Solution projection between refined/coarsened elements
    - Low-storage RK time integration
    
    Attributes:
        nop (int): Polynomial order for the DG basis functions
        ngl (int): Number of LGL points per element (nop + 1)
        nelem (int): Current number of elements in mesh
        xelem (array): Element boundary coordinates
        max_level (int): Maximum allowed refinement level
        max_elements (int): Maximum allowed number of elements
        dt (float): Current time step size
        time (float): Current simulation time
        icase (int): Test case identifier for initial/exact solutions
        dx_min (float): Minimum element size based on max refinement
        q (array): Current solution vector
        wave_speed (float): Wave propagation speed for the equation
    """
    
    def __init__(self, nop, xelem, max_elements, max_level, courant_max=0.1, icase=1):
        """
        Initialize the DG solver with specified parameters.
        
        Args:
            nop (int): Polynomial order for the DG basis
            xelem (array): Initial element boundary coordinates
            max_elements (int): Maximum allowed number of elements
            max_level (int): Maximum allowed refinement level
            courant_max (float): Maximum allowed Courant number for timestep
            icase (int): Test case identifier for initial/exact solutions
        """
        # Store basic solver parameters
        self.nop = nop                    # Polynomial order
        self.xelem = xelem                # Element boundaries
        self.max_elements = max_elements  # Max allowed elements
        self.ngl = nop + 1                # Number of LGL points per element
        self.nelem = len(xelem) - 1       # Initial number of elements
        self.max_level = max_level        # Max refinement level
        self.icase = icase                # Test case number
        self.time = 0.0                   # Initial time

        # Calculate minimum allowed element size based on max refinement
        self.dx_min = np.min(np.diff(xelem)) / (2**max_level)
        
        # Generate LGL quadrature points and weights for solution representation
        self.xgl, self.wgl = lgl_gen(self.ngl)
        
        # Generate quadrature points for exact integration
        # Using nop+2 points ensures exact integration of mass matrices
        self.nq = self.nop + 2  
        self.xnq, self.wnq = lgl_gen(self.nq)
        
        # Create Lagrange basis functions and their derivatives
        # psi: Basis functions evaluated at quadrature points
        # dpsi: Derivatives of basis functions at quadrature points
        self.psi, self.dpsi = Lagrange_basis(self.ngl, self.nq, self.xgl, self.xnq)
        
        # Initialize mesh structures including AMR hierarchy
        self._initialize_mesh()
        
        # Initialize solution vector with initial condition
        self.q = self._initialize_solution()
        
        # Compute stable timestep based on CFL condition
        self._compute_timestep(courant_max)
        
        # Initialize projection matrices for AMR operations
        self._initialize_projections()
        
    def _initialize_mesh(self):
        """
        Initialize mesh structures and grid for computation.
        
        This method:
        1. Computes number of points for both CG and DG representations
        2. Creates hierarchical mesh structure for AMR
        3. Generates computational grid with element connectivity
        """
        # Calculate number of points for different representations
        self.npoin_cg = self.nop * self.nelem + 1  # Points for continuous grid
        self.npoin_dg = self.ngl * self.nelem      # Points for discontinuous grid
        
        # Create hierarchical mesh structure for AMR
        # label_mat: Element labels for refinement hierarchy
        # info_mat: Element information matrix
        # active: Boolean array indicating active elements
        self.label_mat, self.info_mat, self.active = forest(self.xelem, self.max_level)
        
        # Generate computational grid
        # coord: Physical coordinates of all grid points
        # intma: Element connectivity matrix
        # periodicity: Array indicating periodic node connections
        self.coord, self.intma, self.periodicity = create_grid_us(
            self.ngl, self.nelem, self.npoin_cg, self.npoin_dg, 
            self.xgl, self.xelem
        )
        
    def _initialize_solution(self):
        """
        Initialize solution vector using exact solution.
        
        Returns:
            array: Initial solution vector at t=0
            float: Wave propagation speed
        """
        # Get initial condition and wave speed from exact solution
        q, self.wave_speed = exact_solution(
            self.coord, self.npoin_dg, self.time, self.icase
        )
        return q
        
    def _compute_timestep(self, courant_max):
        """
        Compute stable timestep based on CFL condition.
        
        Args:
            courant_max (float): Maximum allowed Courant number
        """
        # Calculate minimum element size accounting for refinement
        dx_min = np.min(np.diff(self.xelem)) / (2**self.max_level)
        
        # Compute timestep: dt = CFL * dx / wave_speed
        self.dt = courant_max * dx_min / self.wave_speed
        
    def _initialize_projections(self):
        """
        Initialize projection matrices for AMR operations.
        
        Creates matrices needed for:
        - Solution projection during h-refinement
        - Solution restriction during h-coarsening
        """
        # Create reference mass matrix for projections
        RM = create_RM_matrix(self.ngl, self.nq, self.wnq, self.psi)
        
        # Create projection matrices:
        # PS1, PS2: Splitting matrices for refinement
        # PG1, PG2: Gathering matrices for coarsening
        self.PS1, self.PS2, self.PG1, self.PG2 = projections(
            RM, self.ngl, self.nq, self.wnq, self.xgl, self.xnq
        )
        
    def _update_matrices(self):
        """
        Update system matrices after mesh adaptation.
        
        Reconstructs:
        1. Mass and stiffness matrices
        2. Global assembly matrices
        3. Flux matrices
        4. Spatial operator
        """
        # Create element-level mass and stiffness matrices
        self.Me = create_mass_matrix(
            self.intma, self.coord, self.nelem, self.ngl, 
            self.nq, self.wnq, self.psi
        )
        self.De = create_diff_matrix(self.ngl, self.nq, self.wnq, self.psi, self.dpsi)
        
        # Assemble global matrices using Direct Stiffness Summation
        self.M, self.D = Matrix_DSS(
            self.Me, self.De, self.wave_speed, self.intma, 
            self.periodicity, self.ngl, self.nelem, self.npoin_dg
        )
        
        # Create upwind flux matrix for interface treatment
        self.F = Fmatrix_upwind_flux(
            self.intma, self.nelem, self.npoin_dg, self.ngl, self.wave_speed
        )
        
        # Compute spatial operator R = D - F
        R = self.D - self.F
        
        # Create final spatial operator Dhat = M^(-1)R
        self.Dhat = np.linalg.solve(self.M, R)
        
    def adapt_mesh(self, criterion=1, marks_override=None):
        """
        Perform mesh adaptation based on solution properties.
        
        This method:
        1. Marks elements for refinement/coarsening
        2. Adapts the mesh structure
        3. Projects the solution to the new mesh
        4. Updates all necessary matrices
        
        Args:
            criterion (int): AMR marking criterion selector
            marks_override (array, optional): Override automatic marking
        """
        # Get refinement marks based on solution properties
        marks = mark(self.active, self.label_mat, self.intma, self.q, criterion)
        
        # Store pre-adaptation state
        pre_grid = self.xelem
        pre_active = self.active
        pre_nelem = self.nelem
        pre_coord = self.coord
        pre_npoin_dg = self.npoin_dg
        
        # Adapt mesh based on marks
        new_grid, new_active, _, new_nelem, npoin_cg, new_npoin_dg = adapt_mesh(
            self.nop, pre_grid, pre_active, self.label_mat, 
            self.info_mat, marks
        )
        
        # Create new computational grid
        new_coord, new_intma, new_periodicity = create_grid_us(
            self.ngl, new_nelem, npoin_cg, new_npoin_dg, 
            self.xgl, new_grid
        )
        
        # Project solution to new mesh
        q_new = adapt_sol(
            self.q, pre_coord, marks, pre_active, self.label_mat,
            self.PS1, self.PS2, self.PG1, self.PG2, self.ngl
        )
        
        # Update solver state with new mesh and solution
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
        
        Implements a 5-stage, 4th-order low-storage Runge-Kutta method
        optimized for wave propagation problems.
        
        Args:
            dt (float, optional): Time step size. Uses self.dt if None.
        """
        if dt is None:
            dt = self.dt
            
        # RK coefficients for low-storage method
        # These coefficients are optimized for wave propagation
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
        dq = np.zeros(self.npoin_dg)  # Stage update
        qp = self.q.copy()            # Working solution copy
        
        # Loop over RK stages
        for s in range(len(RKA)):
            # Compute RHS of wave equation
            R = self.Dhat @ qp
            
            # Update stage values using low-storage formulation
            for i in range(self.npoin_dg):
                dq[i] = RKA[s]*dq[i] + dt*R[i]
                qp[i] = qp[i] + RKB[s]*dq[i]
                
            # Apply periodic boundary conditions if needed
            if self.periodicity[-1] == self.periodicity[0]:
                qp[-1] = qp[0]
                
        # Update solution and time
        self.q = qp
        self.time += dt
        
    def solve(self, time_final):
        """
        Solve wave equation to specified final time.
        
        This method:
        1. Advances solution in time using RK integration
        2. Performs mesh adaptation at each step
        3. Stores solution history
        
        Args:
            time_final (float): Final simulation time
            
        Returns:
            tuple: (times, solutions, grids, coords)
                times: Array of time points
                solutions: List of solution vectors at each time
                grids: List of element boundaries at each time
                coords: List of node coordinates at each time
        """
        # Initialize storage for solution history
        times = [self.time]
        solutions = [self.q.copy()]
        grids = [self.xelem.copy()]
        coords = [self.coord.copy()]
        
        # Time stepping loop
        while self.time < time_final:
            # Adjust final step to hit time_final exactly
            dt = min(self.dt, time_final - self.time)
            
            # Adapt mesh based on solution properties
            self.adapt_mesh()
            
            # Take time step
            self.step(dt)
            
            # Store current state
            times.append(self.time)
            solutions.append(self.q.copy())
            grids.append(self.xelem.copy())
            coords.append(self.coord.copy())
            
        return times, solutions, grids, coords

    def get_exact_solution(self):
        """
        Get exact solution at current time.
        
        Returns:
            array: Exact solution evaluated at current grid points
        """
        qe, _ = exact_solution(self.coord, self.npoin_dg, self.time, self.icase)
        return qe
    
    def reset(self):
        """
        Reset solver to initial state.
        
        This method:
        1. Reinitializes mesh to original configuration
        2. Resets solution to initial
        """

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