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
from ..amr.forest import forest
from ..amr.adapt import adapt_mesh, adapt_sol, mark
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
        self.nop = nop
        self.xelem = xelem
        self.max_elements = max_elements 
        self.ngl = nop + 1
        self.nelem = len(xelem) - 1
        self.max_level = max_level
        self.icase = icase
        self.time = 0.0
        self.dx_min = np.min(np.diff(xelem)) / (2**max_level)
        self.xgl, self.wgl = lgl_gen(self.ngl)
        self.nq = self.nop + 2
        self.xnq, self.wnq = lgl_gen(self.nq)
        self.psi, self.dpsi = Lagrange_basis(self.ngl, self.nq, self.xgl, self.xnq)
        self._initialize_mesh()
        self.q = self._initialize_solution()
        self._compute_timestep(courant_max)
        self._initialize_projections()
        
    def _initialize_mesh(self):
        self.npoin_cg = self.nop * self.nelem + 1
        self.npoin_dg = self.ngl * self.nelem
        self.label_mat, self.info_mat, self.active = forest(self.xelem, self.max_level)
        self.coord, self.intma, self.periodicity = create_grid_us(
            self.ngl, self.nelem, self.npoin_cg, self.npoin_dg, 
            self.xgl, self.xelem
        )
        
    def _initialize_solution(self):
        q, self.wave_speed = exact_solution(
            self.coord, self.npoin_dg, self.time, self.icase
        )
        return q
        
    def _compute_timestep(self, courant_max):
        dx_min = np.min(np.diff(self.xelem)) / (2**self.max_level)
        self.dt = courant_max * dx_min / self.wave_speed
        
    def _initialize_projections(self):
        RM = create_RM_matrix(self.ngl, self.nq, self.wnq, self.psi)
        self.PS1, self.PS2, self.PG1, self.PG2 = projections(
            RM, self.ngl, self.nq, self.wnq, self.xgl, self.xnq
        )
        
    def _update_matrices(self):
        self.Me = create_mass_matrix(
            self.intma, self.coord, self.nelem, self.ngl, 
            self.nq, self.wnq, self.psi
        )
        self.De = create_diff_matrix(self.ngl, self.nq, self.wnq, self.psi, self.dpsi)
        self.M, self.D = Matrix_DSS(
            self.Me, self.De, self.wave_speed, self.intma, 
            self.periodicity, self.ngl, self.nelem, self.npoin_dg
        )
        self.F = Fmatrix_upwind_flux(
            self.intma, self.nelem, self.npoin_dg, self.ngl, self.wave_speed
        )
        R = self.D - self.F
        self.Dhat = np.linalg.solve(self.M, R)
        
    def check_size_ratios(self, element_idx: int, action: int) -> bool:
        elem = self.active[element_idx]
        curr_size = self.info_mat[elem-1][4] - self.info_mat[elem-1][3]
        
        neighbors = []
        for i in range(max(1, elem-2), elem):
            idx = np.where(self.active == i)[0]
            if len(idx) > 0:
                neighbors.append(i)
        for i in range(elem+1, min(elem+3, len(self.label_mat)+1)):
            idx = np.where(self.active == i)[0]
            if len(idx) > 0:
                neighbors.append(i)
                
        if action == 1:
            new_size = curr_size / 2
            for neighbor in neighbors:
                neighbor_size = self.info_mat[neighbor-1][4] - self.info_mat[neighbor-1][3]
                if neighbor_size/new_size > 2:
                    return False
                if neighbor_size/new_size > 2:
                    return False
                    
        elif action == -1:
            new_size = curr_size * 2
            for neighbor in neighbors:
                neighbor_size = self.info_mat[neighbor-1][4] - self.info_mat[neighbor-1][3]
                if new_size/neighbor_size > 2:
                    return False
                if new_size/neighbor_size > 2:
                    return False
        
        return True
  
    def adapt_mesh(self, criterion=1, marks_override=None, element_budget=None):
        """
        Perform mesh adaptation based on solution properties.
        Respects element_budget constraint.
        """
        # Get refinement marks based on solution properties
        marks = mark(self.active, self.label_mat, self.intma, self.q, criterion)

        if marks_override is not None:
            for idx, mark_val in marks_override.items():
                # Check budget before refinement
                if mark_val == 1 and element_budget is not None:
                    if len(self.active) >= element_budget:
                        print(f"Budget limit reached ({element_budget} elements). Canceling refinement.")
                        marks[idx] = 0
                        continue
                marks[idx] = mark_val

        # Enforce 2:1 balance before adaptation
        
        
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

        # Create new grid
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
        
        # Update matrices
        self._update_matrices()

    # def step(self, dt=None):
    #     if dt is None:
    #         dt = self.dt
        
    #         # Check balance before step
    #     violations = check_2_1_balance(self.active, self.label_mat)
    #     if violations:
    #         print(f"\nBalance violations found BEFORE time step at t={self.time}:")
    #         for elem, neighbor, level1, level2 in violations:
    #             print(f"Elements {elem}({level1}) and {neighbor}({level2})")
    #         # print_mesh_state(self.active, self.label_mat)
            
    #     RKA = np.array([0,
    #                    -567301805773.0/1357537059087,
    #                    -2404267990393.0/2016746695238,
    #                    -3550918686646.0/2091501179385,
    #                    -1275806237668.0/842570457699])
        
    #     RKB = np.array([1432997174477.0/9575080441755,
    #                    5161836677717.0/13612068292357,
    #                    1720146321549.0/2090206949498,
    #                    3134564353537.0/4481467310338,
    #                    2277821191437.0/14882151754819])
        
    #     dq = np.zeros(self.npoin_dg)
    #     qp = self.q.copy()
        
    #     for s in range(len(RKA)):
    #         R = self.Dhat @ qp
            
    #         for i in range(self.npoin_dg):
    #             dq[i] = RKA[s]*dq[i] + dt*R[i]
    #             qp[i] = qp[i] + RKB[s]*dq[i]
                
    #         if self.periodicity[-1] == self.periodicity[0]:
    #             qp[-1] = qp[0]
                
    #     self.q = qp
    #     self.time += dt

    #         # Check balance after step
    #     violations = check_2_1_balance(self.active, self.label_mat)
    #     if violations:
    #         print(f"\nBalance violations found AFTER time step at t={self.time}:")
    #         for elem, neighbor, level1, level2 in violations:
    #             print(f"Elements {elem}({level1}) and {neighbor}({level2})")
    #         # print_mesh_state(self.active, self.label_mat)
    def step(self, dt=None):
        """
        Take single time step with balance verification.
        """
        if dt is None:
            dt = self.dt
                
        # Check balance before step
        
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
        
        dq = np.zeros(self.npoin_dg)
        qp = self.q.copy()
        
        for s in range(len(RKA)):
            R = self.Dhat @ qp
            
            for i in range(self.npoin_dg):
                dq[i] = RKA[s]*dq[i] + dt*R[i]
                qp[i] = qp[i] + RKB[s]*dq[i]
                
            if self.periodicity[-1] == self.periodicity[0]:
                qp[-1] = qp[0]
                
        self.q = qp
        self.time += dt

        # Check balance after step  

    def solve(self, time_final):
        times = [self.time]
        solutions = [self.q.copy()]
        grids = [self.xelem.copy()]
        coords = [self.coord.copy()]
        
        step_count = 0
        while self.time < time_final:
            dt = min(self.dt, time_final - self.time)
            print(f"\nTimestep {step_count}, Time: {self.time:.3f}")
            
            # Single adapt_mesh call 
            self.adapt_mesh()
            
            # Take time step
            self.step(dt)
            
            # Store results
            times.append(self.time)
            solutions.append(self.q.copy())
            grids.append(self.xelem.copy())
            coords.append(self.coord.copy())
            step_count += 1
            
        return times, solutions, grids, coords 
    # def solve(self, time_final):
    #     times = [self.time]
    #     solutions = [self.q.copy()]
    #     grids = [self.xelem.copy()]
    #     coords = [self.coord.copy()]
        
    #     step_count = 0
    #     while self.time < time_final:
    #         dt = min(self.dt, time_final - self.time)
            
    #         print(f"\nTimestep {step_count}, Time: {self.time:.3f}")
        
    #         # Check balance before adaptation
    #         violations = check_2_1_balance(self.active, self.label_mat, debug=True)
    #         if violations:
    #             print("Pre-adaptation violations found:")
    #             for elem, neighbor, level1, level2 in violations:
    #                 print(f"Elements {elem}({level1}) and {neighbor}({level2})")

    #         self.adapt_mesh()

    #                 # Check balance after adaptation
    #         violations = check_2_1_balance(self.active, self.label_mat, debug=True)
    #         if violations:
    #             print("Post-adaptation violations found:")
    #             for elem, neighbor, level1, level2 in violations:
    #                 print(f"Elements {elem}({level1}) and {neighbor}({level2})")
                
    #             # Print detailed mesh state when violation found
    #             print("\nDetailed mesh state at violation:")
    #             # print_mesh_state(self.active, self.label_mat)
            
    #         self.step(dt)
            
    #         times.append(self.time)
    #         solutions.append(self.q.copy())
    #         grids.append(self.xelem.copy())
    #         coords.append(self.coord.copy())
    #         step_count += 1
            
    #     return times, solutions, grids, coords

    def get_exact_solution(self):
        qe, _ = exact_solution(self.coord, self.npoin_dg, self.time, self.icase)
        return qe
    
    def reset(self):
        self.nelem = len(self.xelem) - 1
        self.npoin_cg = self.nop * self.nelem + 1
        self.npoin_dg = self.ngl * self.nelem
        self.label_mat, self.info_mat, self.active = forest(self.xelem, self.max_level)
        self.coord, self.intma, self.periodicity = create_grid_us(
            self.ngl, self.nelem, self.npoin_cg, self.npoin_dg, self.xgl, self.xelem
        )
        self.q, _ = exact_solution(self.coord, self.npoin_dg, 0.0, self.icase)
        self.time = 0.0
        self._update_matrices()
        return self.q
