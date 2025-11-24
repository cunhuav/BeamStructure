# src/nonlinear_solver.py
import numpy as np

class NonlinearSolver:
    def __init__(self, max_iter=20, tol=1e-6, verbose=True):
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

    def solve(self, model, F_ext, fixed_dofs, U0=None):
        """
        model: any object with methods:
            - compute_internal_force(U) -> F_int
            - compute_tangent_stiffness(U) -> K_tan
        """
        ndof = len(F_ext)
        if U0 is None:
            U = np.zeros(ndof)
        else:
            U = U0.copy()
        
        for it in range(self.max_iter):
            F_int = model.compute_internal_force(U)
            R = F_ext - F_int
            
            # Apply BCs to residual
            for dof in fixed_dofs:
                R[dof] = 0.0
            
            res_norm = np.linalg.norm(R)
            if self.verbose:
                print(f"Iter {it+1}: ||R|| = {res_norm:.3e}")
            if res_norm < self.tol:
                print("✅ Converged!")
                return U
            
            K_tan = model.compute_tangent_stiffness(U)
            
            # Apply BCs to tangent stiffness
            K_tan_bc = K_tan.copy()
            for dof in fixed_dofs:
                K_tan_bc[dof, :] = 0.0
                K_tan_bc[:, dof] = 0.0
                K_tan_bc[dof, dof] = 1.0
            
            dU = np.linalg.solve(K_tan_bc, R)
            U += dU
        
        raise RuntimeError("Newton-Raphson failed to converge")