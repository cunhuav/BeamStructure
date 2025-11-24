# src/linear_beam.py
import numpy as np
import matplotlib.pyplot as plt

def beam_element_stiffness(L, E, A, I):
    """
    Compute the 6x6 stiffness matrix for an Euler-Bernoulli beam element.
    
    DOF order: [u1, w1, theta1, u2, w2, theta2]
    
    Parameters:
    ----------
    L : float
        Element length
    E : float
        Young's modulus
    A : float
        Cross-sectional area
    I : float
        Second moment of area (area moment of inertia)
        
    Returns:
    -------
    k : np.ndarray (6, 6)
        Element stiffness matrix
    """
    k = np.zeros((6, 6))
    
    # Axial terms
    k[0, 0] =  E * A / L
    k[0, 3] = -E * A / L
    k[3, 0] = -E * A / L
    k[3, 3] =  E * A / L
    
    # Bending terms
    k[1, 1] =  12 * E * I / L**3
    k[1, 2] =   6 * E * I / L**2
    k[1, 4] = -12 * E * I / L**3
    k[1, 5] =   6 * E * I / L**2

    k[2, 1] =   6 * E * I / L**2
    k[2, 2] =   4 * E * I / L
    k[2, 4] =  -6 * E * I / L**2
    k[2, 5] =   2 * E * I / L

    k[4, 1] = -12 * E * I / L**3
    k[4, 2] =  -6 * E * I / L**2
    k[4, 4] =  12 * E * I / L**3
    k[4, 5] =  -6 * E * I / L**2

    k[5, 1] =   6 * E * I / L**2
    k[5, 2] =   2 * E * I / L
    k[5, 4] =  -6 * E * I / L**2
    k[5, 5] =   4 * E * I / L

    # Symmetrize (for numerical safety)
    k = (k + k.T) / 2.0
    return k


def assemble_global_stiffness(elements, nodes, E, A, I):
    """
    Assemble global stiffness matrix.
    
    Parameters:
    ----------
    elements : list of [node_i, node_j]
        Element connectivity
    nodes : np.ndarray (n_nodes, 2)
        Nodal coordinates [x, y] (only x used)
    E, A, I : material and section properties (assumed constant)
        
    Returns:
    -------
    K : np.ndarray (3*n_nodes, 3*n_nodes)
    """
    n_nodes = len(nodes)
    ndof = 3 * n_nodes
    K = np.zeros((ndof, ndof))
    
    for el_idx, (n1, n2) in enumerate(elements):
        x1, _ = nodes[n1]
        x2, _ = nodes[n2]
        L = abs(x2 - x1)
        k_local = beam_element_stiffness(L, E, A, I)
        
        # Global DOF indices for this element
        dof_map = [3*n1, 3*n1+1, 3*n1+2, 3*n2, 3*n2+1, 3*n2+2]
        
        # Assemble
        for i in range(6):
            for j in range(6):
                K[dof_map[i], dof_map[j]] += k_local[i, j]
                
    return K

def assemble_global_matrix(elements, nodes, element_matrix_func):
    """
    Generic assembler: call element_matrix_func(L, elem_idx, n1, n2) for each element.
    """
    n_nodes = len(nodes)
    ndof = 3 * n_nodes
    K = np.zeros((ndof, ndof))
    for el_idx, (n1, n2) in enumerate(elements):
        x1, _ = nodes[n1]
        x2, _ = nodes[n2]
        L = abs(x2 - x1)
        k_local = element_matrix_func(L=L, elem_idx=el_idx, n1=n1, n2=n2)
        dof_map = [3*n1, 3*n1+1, 3*n1+2, 3*n2, 3*n2+1, 3*n2+2]
        for i in range(6):
            for j in range(6):
                K[dof_map[i], dof_map[j]] += k_local[i, j]
    return K

def apply_boundary_conditions(K, F, fixed_dofs):
    """
    Apply homogeneous Dirichlet BCs by modifying K and F.
    """
    K_bc = K.copy()
    F_bc = F.copy()
    
    for dof in fixed_dofs:
        K_bc[dof, :] = 0.0
        K_bc[:, dof] = 0.0
        K_bc[dof, dof] = 1.0
        F_bc[dof] = 0.0
        
    return K_bc, F_bc


def solve_cantilever_tip_load():
    """
    Example: Cantilever beam with tip vertical load.
    - Length L = 1.0 m
    - E = 210e9 Pa, A = 0.01 m², I = 8.333e-6 m⁴ (rectangular 0.1x0.1 m)
    - Tip load P = -1000 N at free end
    """
    # Beam properties
    L = 1.0
    E = 210e9
    b = h = 0.1
    A = b * h
    I = b * h**3 / 12  # = 8.333e-6 m⁴
    
    # Discretization
    n_elements = 10
    n_nodes = n_elements + 1
    x_coords = np.linspace(0, L, n_nodes)
    nodes = np.column_stack([x_coords, np.zeros(n_nodes)])
    elements = [[i, i+1] for i in range(n_elements)]
    
    # Global system
    ndof = 3 * n_nodes
    K = assemble_global_stiffness(elements, nodes, E, A, I)
    F = np.zeros(ndof)
    
    # Apply tip load: vertical force at last node
    tip_node = n_nodes - 1
    F[3*tip_node + 1] = -1000.0  # w-direction (DOF index 1 in node)
    
    # Boundary conditions: clamped at node 0
    fixed_dofs = [0, 1, 2]  # u, w, theta fixed at left end
    
    # Solve
    K_bc, F_bc = apply_boundary_conditions(K, F, fixed_dofs)
    U = np.linalg.solve(K_bc, F_bc)
    
    # Extract displacements
    w = U[1::3]  # w displacements (every 3rd starting at index 1)
    
    # Analytical solution for comparison
    x = x_coords
    w_exact = (-1000.0) * (x**2) * (3*L - x) / (6 * E * I)
    
    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot(x, w, 'bo-', label='FEM (10 elements)')
    plt.plot(x, w_exact, 'r--', label='Analytical')
    plt.xlabel('x (m)')
    plt.ylabel('Transverse displacement w (m)')
    plt.title('Cantilever Beam with Tip Load')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    print(f"Tip displacement (FEM): {w[-1]:.6e} m")
    print(f"Tip displacement (Exact): {w_exact[-1]:.6e} m")
    print(f"Error: {abs(w[-1] - w_exact[-1]) / abs(w_exact[-1]) * 100:.2f}%")
    
    return U

def solve_cantilever_axial_and_transverse_load():
    """
    Example: Cantilever beam with tip axial tension P and transverse load Q.
    
    - Length L = 1.0 m
    - Rectangular cross-section: b = h = 0.1 m
    - E = 210e9 Pa
    - Tip loads: 
        * Axial tension P = 10,000 N (positive = tension)
        * Transverse load Q = -1,000 N (negative = downward)
    """
    # Beam properties
    L = 1.0
    E = 210e9
    b = h = 0.1
    A = b * h
    I = b * h**3 / 12  # = 8.333e-6 m⁴
    
    # Loads
    P = 10000.0   # Axial tension (N)
    Q = -1000.0   # Transverse tip load (N)
    
    # Discretization
    n_elements = 10
    n_nodes = n_elements + 1
    x_coords = np.linspace(0, L, n_nodes)
    nodes = np.column_stack([x_coords, np.zeros(n_nodes)])
    elements = [[i, i+1] for i in range(n_elements)]
    
    # Global system
    ndof = 3 * n_nodes
    K = assemble_global_stiffness(elements, nodes, E, A, I)
    F = np.zeros(ndof)
    
    # Apply tip loads at last node
    tip_node = n_nodes - 1
    F[3*tip_node + 0] = P   # u-direction (axial)
    F[3*tip_node + 1] = Q   # w-direction (transverse)
    # Moment = 0 → no load on theta DOF
    
    # Boundary conditions: clamped at node 0 (u=0, w=0, theta=0)
    fixed_dofs = [0, 1, 2]
    
    # Solve
    K_bc, F_bc = apply_boundary_conditions(K, F, fixed_dofs)
    U = np.linalg.solve(K_bc, F_bc)
    
    # Extract displacements
    u = U[0::3]   # axial displacement
    w = U[1::3]   # transverse displacement
    
    # Analytical solutions
    x = x_coords
    u_exact = (P / (E * A)) * x
    w_exact = Q * (x**2) * (3*L - x) / (6 * E * I)
    
    # Plot axial displacement
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(x, u, 'bo-', label='FEM')
    plt.plot(x, u_exact, 'r--', label='Analytical')
    plt.xlabel('x (m)')
    plt.ylabel('Axial displacement u (m)')
    plt.title('Axial Displacement')
    plt.legend()
    plt.grid(True)
    
    # Plot transverse displacement
    plt.subplot(1, 2, 2)
    plt.plot(x, w, 'bo-', label='FEM')
    plt.plot(x, w_exact, 'r--', label='Analytical')
    plt.xlabel('x (m)')
    plt.ylabel('Transverse displacement w (m)')
    plt.title('Transverse Displacement')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print tip results
    print(f"Tip axial displacement (FEM):   {u[-1]:.6e} m")
    print(f"Tip axial displacement (Exact): {u_exact[-1]:.6e} m")
    print(f"Tip axial error: {abs(u[-1] - u_exact[-1]) / abs(u_exact[-1]) * 100:.2f}%\n")
    
    print(f"Tip transverse displacement (FEM):   {w[-1]:.6e} m")
    print(f"Tip transverse displacement (Exact): {w_exact[-1]:.6e} m")
    print(f"Tip transverse error: {abs(w[-1] - w_exact[-1]) / abs(w_exact[-1]) * 100:.2f}%")
    
    return U



if __name__ == "__main__":
    solve_cantilever_tip_load()