# BeamStructureFEM: A Finite Element Framework for Euler-Bernoulli Beam Analysis

> **Project Goal**: Implement a modular, educational finite element code for Euler-Bernoulli beams under increasing levels of nonlinearity:  
> 1. **Linear elastic** (small deformation, linear material)  
> 2. **Material nonlinearity** (small deformation, nonlinear stress-strain)  
> 3. **Geometric nonlinearity** (large displacement, small rotation – von Kármán assumption)  
>
> All derivations use Unicode math symbols for GitHub compatibility. Code in **Python** (NumPy/SciPy).

---

## 📚 Theory & Formulation

### 1. Linear Elastic Euler-Bernoulli Beam (Small Deformation)

#### Assumptions
- Plane sections remain plane and perpendicular to the neutral axis.
- Small displacements and rotations: `|w| ≪ L`, `|θ| ≪ 1` rad.
- Linear elastic isotropic material: `σ = E ε`.

#### Kinematics
- Transverse displacement: `w(x)`
- Axial strain:  
  `εₓₓ(x, y) = du₀/dx − y · d²w/dx²`  
  where `u₀(x)` = axial displacement of neutral axis, `y` = distance from neutral axis.

#### Weak Form (Principle of Virtual Work)
```
δΠ = ∫₀ᴸ [ EA · (du₀/dx) · δ(du₀/dx) + EI · (d²w/dx²) · δ(d²w/dx²) ] dx
     − ∫₀ᴸ q · δw dx − Σ Pᵢ · δwᵢ = 0
```

#### Element Stiffness Matrix (2-Node Beam Element)
Degrees of freedom per node: `[u, w, θ]ᵀ`

```
kᵉ =
[ EA/L      0           0         -EA/L      0           0        ]
[ 0         12EI/L³     6EI/L²     0        -12EI/L³     6EI/L²   ]
[ 0         6EI/L²      4EI/L      0        -6EI/L²      2EI/L    ]
[ -EA/L     0           0          EA/L      0           0        ]
[ 0        -12EI/L³    -6EI/L²     0         12EI/L³    -6EI/L²   ]
[ 0         6EI/L²      2EI/L      0        -6EI/L²      4EI/L    ]
```

---

### 2. Material Nonlinearity (Small Deformation, Nonlinear σ–ε)

#### Assumptions
- Same kinematics as linear case (`|w| ≪ L`, `|θ| ≪ 1`).
- Nonlinear uniaxial stress-strain law, e.g.:  
  - Ramberg-Osgood: `ε = σ/E + 0.002 · (σ/σ₀.₂)ⁿ`  
  - Bilinear elastoplastic:  
    `σ = E·ε` if `|σ| < σ_y`,  
    else `σ = σ_y + E_t · (ε − ε_y)`

#### Constitutive Relation
`σₓₓ(x, y) = f(εₓₓ(x, y))`  (nonlinear function)

#### Weak Form
```
δΠ = ∫₀ᴸ ∫_A σₓₓ · δεₓₓ dA dx − external work = 0
```

#### Solution: Newton-Raphson Iteration
- Internal force vector:  
  `fⁱⁿᵗ = ∫₀ᴸ Bᵀ · σ dx`
- Tangent stiffness matrix:  
  `Kᵗᵃⁿ = ∫₀ᴸ Bᵀ · Dᵗᵃⁿ · B dx`  
  where `Dᵗᵃⁿ = dσ/dε` (material tangent modulus)

> **Note**: Cross-section integration via Gauss quadrature to handle `σ(y)` nonlinearity.

---

### 3. Geometric Nonlinearity: Large Displacement, Small Rotation (von Kármán)

#### Assumptions
- **Large `w(x)`**, but **small slope**: `|dw/dx| ≪ 1`
- Includes nonlinear stretching from bending

#### Kinematics (von Kármán Strain)
```
εₓₓ = du₀/dx + ½ · (dw/dx)² − y · d²w/dx²
```

#### Weak Form
```
δΠ = ∫₀ᴸ [ N · (δu₀' + w' · δw') + M · δw'' ] dx − ∫₀ᴸ q · δw dx = 0
```
where  
- `N = ∫_A σₓₓ dA` (axial force)  
- `M = −∫_A y · σₓₓ dA` (bending moment)

#### Tangent Stiffness (Newton-Raphson)
```
Kᵗᵃⁿ = K_L + K_NL(u)
```
- `K_L`: Linear stiffness (bending + axial)
- `K_NL`: Geometric stiffness (depends on current displacement)

Geometric stiffness matrix for a beam element (with axial force `N`):
```
K_NL = (N / (30·L)) ·
[ 0   0    0    0   0    0  ]
[ 0  36   3L    0  -36   3L ]
[ 0  3L  4L²    0  -3L  -L² ]
[ 0   0    0    0   0    0  ]
[ 0 -36  -3L    0   36  -3L ]
[ 0  3L  -L²    0  -3L  4L² ]
```

> **Note**: `N` must be updated each iteration from current `w(x)`.

---

## 🛠️ Implementation Plan

| Stage | Features |
|------|--------|
| v0.1 | Linear elastic Euler beam (2D) |
| v0.2 | Material nonlinearity (user-defined σ–ε) |
| v0.3 | von Kármán geometric nonlinearity |
| v1.0 | Full pipeline + examples + visualization |

---

## 📖 References
1. Cook et al. *Concepts and Applications of Finite Element Analysis*  
2. Bathe. *Finite Element Procedures*  
3. Reddy. *An Introduction to Nonlinear Finite Element Analysis*
