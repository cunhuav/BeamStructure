# src/material.py
import numpy as np
import matplotlib.pyplot as plt

class BilinearMaterial:
    """
    Bilinear elastoplastic material model for 1D stress state.
    
    Distinguishes:
      - epsilon_p: signed plastic strain (can be + or -)
      - epsilon_p_bar: accumulated (equivalent) plastic strain (>= 0, non-decreasing)
    
    Hardening is driven by epsilon_p_bar.
    """
    
    def __init__(self, E, sigma_y0, H=0.0):
        """
        Parameters:
        ----------
        E : float
            Elastic modulus (Pa)
        sigma_y0 : float
            Initial yield stress (Pa)
        H : float
            Plastic hardening modulus (dσ_y / dε̄^p), >= 0 (Pa)
        """
        self.E = E
        self.sigma_y0 = sigma_y0
        self.H = H
        self.state = 'elastic'  # or 'plastic'
        self.reset()
        
    def reset(self):
        """Reset internal state variables."""
        self.epsilon_p = 0.0      # signed plastic strain
        self.epsilon_p_bar = 0.0  # accumulated plastic strain (>= 0)
        self.sigma = 0.0
        self.E_tan = self.E
        self.state = 'elastic'
        
    def update_state(self, epsilon):
        """
        Update stress and tangent modulus given total strain ε.
        
        Implements 1D associative elastoplasticity with isotropic hardening.
        
        Parameters:
        ----------
        epsilon : float
            Total axial strain (ε = ε_e + ε_p)
            
        Returns:
        -------
        sigma : float
            Updated stress
        E_tan : float
            Updated tangent modulus (dσ/dε)
        """
        # Elastic trial
        epsilon_e_trial = epsilon - self.epsilon_p
        sigma_trial = self.E * epsilon_e_trial
        f_trial = abs(sigma_trial) - (self.sigma_y0 + self.H * self.epsilon_p_bar)
        
        if f_trial <= 0:
            # Elastic loading/unloading
            self.sigma = sigma_trial
            self.E_tan = self.E
            self.state = 'elastic'
        else:
            # Plastic loading
            dlambda = f_trial / (self.E + self.H)
            sign_sig = np.sign(sigma_trial)
            
            # Update signed plastic strain
            self.epsilon_p += dlambda * sign_sig
            
            # Update accumulated plastic strain (always increases)
            self.epsilon_p_bar += dlambda  # because dλ = |dε^p| >= 0
            
            # Update stress using accumulated plastic strain
            self.sigma = sign_sig * (self.sigma_y0 + self.H * self.epsilon_p_bar)
            
            # Update tangent modulus
            self.E_tan = (self.E * self.H) / (self.E + self.H)
            self.state = 'plastic'
            
        return self.sigma, self.E_tan

    def get_stress_and_tangent(self):
        """
        Convenient method for FEM to get (sigma, Et) without dict unpacking.
        """
        return self.sigma, self.E_tan
        
    def is_plastic(self):
        return self.state == 'plastic'
    
    def get_state(self):
        """Return current internal state for debugging."""
        return {
            'epsilon_p': self.epsilon_p,
            'epsilon_p_bar': self.epsilon_p_bar,
            'sigma': self.sigma,
            'E_tan': self.E_tan
        }

# ==============================
# Default test when running this file directly
# ==============================
if __name__ == "__main__":
    print("Running default test for BilinearMaterial...")
    
    # Create material
    mat = BilinearMaterial(E=210e9, sigma_y0=250e6, H=2.1e9)
    
    # Define complex loading path
    eps_path = np.concatenate([
        np.linspace(0, 0.004, 80),      # tension into plasticity
        np.linspace(0.004, -0.002, 100) # unload and compress
    ])
    
    # Record history
    sigma_path = []
    E_tan_path = []
    eps_p_bar_path = []
    
    mat.reset()
    for eps in eps_path:
        sigma, E_tan = mat.update_state(eps)
        sigma_path.append(sigma)
        E_tan_path.append(E_tan)
        eps_p_bar_path.append(mat.epsilon_p_bar)
    
    sigma_path = np.array(sigma_path)
    E_tan_path = np.array(E_tan_path)
    eps_p_bar_path = np.array(eps_p_bar_path)
    
    # Plot
    plt.figure(figsize=(15, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(eps_path, sigma_path / 1e6, 'b-', linewidth=2)
    plt.axhline(250, color='r', linestyle='--', alpha=0.7, label=r'$\sigma_y$')
    plt.axhline(-250, color='r', linestyle='--', alpha=0.7)
    plt.xlabel('Total Strain $\\varepsilon$')
    plt.ylabel('Stress $\\sigma$ (MPa)')
    plt.title('Stress-Strain Response')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(eps_path, E_tan_path / 1e9, 'm-', linewidth=2)
    plt.axhline(210, color='k', linestyle='--', alpha=0.7, label='$E = 210$ GPa')
    E_tan_plastic = (210e9 * 2.1e9) / (210e9 + 2.1e9) / 1e9
    plt.axhline(E_tan_plastic, color='orange', linestyle='--', alpha=0.7,
               label=f'$E^{{\\tan}} = {E_tan_plastic:.2f}$ GPa')
    plt.xlabel('Total Strain $\\varepsilon$')
    plt.ylabel('Tangent Modulus $E^{\\tan}$ (GPa)')
    plt.title('Tangent Modulus Evolution')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(eps_path, eps_p_bar_path, 'g-', linewidth=2)
    plt.xlabel('Total Strain $\\varepsilon$')
    plt.ylabel('Accumulated Plastic Strain $\\bar{\\varepsilon}^p$')
    plt.title('Hardening Variable')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print key values
    eps_y = 250e6 / 210e9
    print(f"Yield strain ε_y = σ_y / E = {eps_y:.6f}")
    
    plastic_idx = np.where(E_tan_path < 200e9)[0]
    if len(plastic_idx) > 0:
        avg_E_tan = np.mean(E_tan_path[plastic_idx]) / 1e9
        print(f"Average plastic tangent modulus: {avg_E_tan:.3f} GPa")
        print(f"Theoretical E^tan = (E*H)/(E+H) = {E_tan_plastic:.3f} GPa")