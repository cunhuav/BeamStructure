# src/material_nonlinear_beam.py
import numpy as np
from linear_beam import assemble_global_matrix, apply_boundary_conditions
from material import BilinearMaterial  # your existing class

class MaterialNonlinearBeamModel:
    def __init__(self, nodes, elements, material_template, b, h, ngp_section=3):
        """
        material_template: a BilinearMaterial instance used as template to create copies.
        """
        self.nodes = nodes
        self.elements = elements
        self.material_template = material_template
        self.b = b
        self.h = h
        self.ngp_section = ngp_section
        # Step 1: Precompute Gauss points (defines beam_gp, y_gp, w_gp)
        self._precompute_gauss()
        self.gp_stress_strain_history = []
        # Step 3: Initialize materials
        self._initialize_materials()  # ← create independent material instances
        

    def _precompute_gauss(self):
        self.beam_gp = [(-1/np.sqrt(3), 1.0), (1/np.sqrt(3), 1.0)]
        if self.ngp_section == 2:
            gp_sec = [-1/np.sqrt(3), 1/np.sqrt(3)]
            gw_sec = [1.0, 1.0]
        elif self.ngp_section == 3:
            gp_sec = [-np.sqrt(3/5), 0.0, np.sqrt(3/5)]
            gw_sec = [5/9, 8/9, 5/9]
        else:
            raise ValueError("ngp_section must be 2 or 3")
        self.y_gp = [yg * (self.h / 2) for yg in gp_sec]
        self.w_gp = [wg * (self.h / 2) for wg in gw_sec]

    def _initialize_materials(self):
        """
        Create independent BilinearMaterial instance for each Gauss point.
        self.materials[elem_idx][beam_gp_idx][section_gp_idx] = BilinearMaterial(...)
        """
        self.materials = []
        for _ in range(len(self.elements)):
            elem_mats = []
            for _ in self.beam_gp:
                section_mats = [
                    BilinearMaterial(
                        E=self.material_template.E,
                        sigma_y0=self.material_template.sigma_y0,
                        H=self.material_template.H
                    )
                    for _ in self.y_gp
                ]
                elem_mats.append(section_mats)
            self.materials.append(elem_mats)

    def _get_B_matrices(self, L, xi):
        B_axial = np.array([-1/L, 0, 0, 1/L, 0, 0])
        d2N1 = (12*xi - 6) / L**2
        d2N2 = (6*xi - 4) / L
        d2N3 = (-12*xi + 6) / L**2
        d2N4 = (6*xi - 2) / L
        B_bending = np.array([0, d2N1, d2N2, 0, d2N3, d2N4])
        return B_axial, B_bending

    def compute_internal_force(self, U):
        """Update all Gauss point states and compute F_int."""
        ndof = len(U)
        F_int = np.zeros(ndof)

        self.gp_plastic_flag = np.zeros((len(self.elements), len(self.beam_gp), len(self.y_gp)), dtype=bool)

        current_step_data = []

        for el_idx, (n1, n2) in enumerate(self.elements):
            x1, _ = self.nodes[n1]
            x2, _ = self.nodes[n2]
            L = abs(x2 - x1)
            dof_map = [3*n1, 3*n1+1, 3*n1+2, 3*n2, 3*n2+1, 3*n2+2]
            d_local = U[dof_map]

            f_int_e = np.zeros(6)
            for i_gp, (xsi, w_b) in enumerate(self.beam_gp):
                xi = (1 + xsi) / 2.0
                J = L / 2.0
                w_x = w_b * J
                B_ax, B_b = self._get_B_matrices(L, xi)
                eps0 = B_ax @ d_local
                kappa = B_b @ d_local

                N = 0.0
                M = 0.0
                gp_data_row = []
                for j_gp, (yg, wg) in enumerate(zip(self.y_gp, self.w_gp)):
                    eps_total = eps0 - yg * kappa
                    mat_gp = self.materials[el_idx][i_gp][j_gp]
                    
                    # 👇 Update the material's internal state
                    mat_gp.update_state(eps_total)
                    sigma, _ = mat_gp.get_stress_and_tangent()
                    
                    dA = self.b * wg
                    N += sigma * dA
                    M -= sigma * yg * dA
                    
                    # 👇 记录该高斯点数据
                    is_plastic = mat_gp.is_plastic()  # 假设你的 BilinearMaterial 有这个方法
                    gp_data_row.append((yg, eps_total, sigma, is_plastic))

                    # 👇 标记塑性
                    self.gp_plastic_flag[el_idx, i_gp, j_gp] = is_plastic      

                f_int_e += (N * B_ax + M * B_b) * w_x

                # 记录本 beam GP 数据
                current_step_data.append({
                    'element': el_idx,
                    'beam_gp': i_gp,
                    'section_gauss_points': gp_data_row
                })
            
            for i, gdof in enumerate(dof_map):
                F_int[gdof] += f_int_e[i]

            # 保存本步数据
            self.gp_stress_strain_history.append(current_step_data)
            
        return F_int

    def compute_tangent_stiffness(self, U):
        """Compute K_tan using CURRENT states (after compute_internal_force)."""
        def element_tangent(L=None, elem_idx=None, n1=None, n2=None):
            dof_map = [3*n1, 3*n1+1, 3*n1+2, 3*n2, 3*n2+1, 3*n2+2]
            d_local = U[dof_map]
            k_tan_e = np.zeros((6, 6))

            for i_gp, (xsi, w_b) in enumerate(self.beam_gp):
                xi = (1 + xsi) / 2.0
                J = L / 2.0
                w_x = w_b * J
                B_ax, B_b = self._get_B_matrices(L, xi)
                eps0 = B_ax @ d_local
                kappa = B_b @ d_local

                EA_tan = 0.0
                EI_tan = 0.0
                for j_gp, (yg, wg) in enumerate(zip(self.y_gp, self.w_gp)):
                    mat_gp = self.materials[elem_idx][i_gp][j_gp]
                    _, Et = mat_gp.get_stress_and_tangent()
                    dA = self.b * wg
                    EA_tan += Et * dA
                    EI_tan += Et * (yg**2) * dA

                k_ax = EA_tan * np.outer(B_ax, B_ax)
                k_bend = EI_tan * np.outer(B_b, B_b)
                k_tan_e += (k_ax + k_bend) * w_x
            return k_tan_e

        return assemble_global_matrix(self.elements, self.nodes, element_tangent)