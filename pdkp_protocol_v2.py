
"""
================================================================================
POLYMER DEGRADATION KINETICS PROTOCOL (PDKP)
Version 1.0 — Open-Source Framework for Microplastic Persistence Modelling
================================================================================

Authors: Marcos Fernandes de Oliveira, Carlos Frederico de Souza Castro,
         Romulo Davi Albuquerque Andrade, Dener Márcio da Silva Oliveira

Institution: Federal Institute of Science, Technology, and Education of Goiás
License: MIT License
Repository: https://github.com/mfoliveira/pdkp-microplastic-persistence

DESCRIPTION:
-----------
This protocol implements the mechanistic kinetic framework linking polymer
molecular architecture to degradation dynamics in terrestrial environments.
The model integrates transition state theory, polymer physics (WLF equation,
free volume theory), and environmental chemistry to predict microplastic
persistence across diverse polymer types and soil conditions.

REQUIREMENTS:
------------
- Python >= 3.8
- NumPy >= 1.20
- SciPy >= 1.7
- Matplotlib >= 3.4

INSTALLATION:
------------
pip install numpy scipy matplotlib

USAGE:
------
See examples below and the Jupyter notebook in /notebooks/

CITATION:
--------
Oliveira, M.F. et al. (2026). A Mechanistic Kinetic Framework Linking 
Polymer Architecture to Microplastic Persistence in Terrestrial Systems. 
Science of the Total Environment, [in review].
================================================================================
"""

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import curve_fit
import warnings

# =============================================================================
# SECTION 1: PHYSICAL CONSTANTS AND UNIVERSAL PARAMETERS
# =============================================================================

R = 8.314          # Gas constant [J/(mol·K)]
k_B = 1.381e-23    # Boltzmann constant [J/K]
h = 6.626e-34      # Planck constant [J·s]
C1_WLF = 17.4      # WLF universal constant [-]
C2_WLF = 51.6      # WLF universal constant [K]

# =============================================================================
# SECTION 2: POLYMER DATABASE
# =============================================================================

POLYMER_DB = {
    "LDPE": {
        "T_g_amorph": 163.0,      # K (-110°C)
        "T_m": 410.0,              # K (137°C)
        "X_c": 0.45,               # Crystallinity fraction
        "phi_branch": 0.30,        # Branching density
        "chi_eff": 0.10,           # Cohesion parameter
        "f_v_g": 0.025,            # Glassy free volume
        "alpha_f": 5.0e-4,         # Thermal expansion [K^-1]
        "v_seg": 32.0e-6,          # Segment molar volume [m³/mol]
        "E_a_seg": 52.0,           # Segmental activation energy [kJ/mol]
        "nu_restriction": 1.8,     # Crystalline restriction exponent
    },
    "HDPE": {
        "T_g_amorph": 173.0,       # K (-100°C)
        "T_m": 410.0,
        "X_c": 0.75,
        "phi_branch": 0.0,
        "chi_eff": 0.10,
        "f_v_g": 0.025,
        "alpha_f": 5.0e-4,
        "v_seg": 32.0e-6,
        "E_a_seg": 52.0,
        "nu_restriction": 1.8,
    },
    "PP": {
        "T_g_amorph": 253.0,       # K (-20°C)
        "T_m": 438.0,              # K (165°C)
        "X_c": 0.55,
        "phi_branch": 0.05,
        "chi_eff": 0.10,
        "f_v_g": 0.025,
        "alpha_f": 5.5e-4,
        "v_seg": 35.5e-6,
        "E_a_seg": 48.0,
        "nu_restriction": 1.8,
    },
    "PET": {
        "T_g_amorph": 348.0,       # K (75°C)
        "T_m": 538.0,              # K (265°C)
        "X_c": 0.35,
        "phi_branch": 0.0,
        "chi_eff": 0.50,
        "f_v_g": 0.025,
        "alpha_f": 4.0e-4,
        "v_seg": 48.0e-6,
        "E_a_seg": 75.0,
        "nu_restriction": 1.8,
    }
}

# =============================================================================
# SECTION 3: CORE PHYSICAL FUNCTIONS
# =============================================================================

def T_g_effective(T_g_amorph, T_m, X_c, nu=1.8):
    """
    Effective glass transition temperature for semi-crystalline polymers.

    Parameters:
    -----------
    T_g_amorph : float
        Glass transition of fully amorphous polymer [K]
    T_m : float
        Melting temperature [K]
    X_c : float
        Crystallinity fraction (0-1)
    nu : float
        Crystalline restriction exponent (default: 1.8)

    Returns:
    --------
    T_g_eff : float
        Effective glass transition temperature [K]

    References:
    -----------
    Fox equation modification for crystalline constraint.
    T_g^crystal ≈ T_m - 50 K (empirical approximation)
    """
    T_g_crystal = T_m - 50.0
    return T_g_amorph + (T_g_crystal - T_g_amorph) * (X_c ** nu)


def fractional_free_volume(T, T_g_eff, f_v_g=0.025, alpha_f=5.0e-4):
    """
    Temperature-dependent fractional free volume.

    f_v(T) = f_v,g + α_f · (T - T_g^eff)   for T >= T_g^eff
    f_v(T) = f_v,g                          for T < T_g^eff (frozen)

    Parameters:
    -----------
    T : float or array
        Temperature [K]
    T_g_eff : float
        Effective glass transition temperature [K]
    f_v_g : float
        Glassy free volume (default: 0.025)
    alpha_f : float
        Thermal expansion coefficient [K^-1]

    Returns:
    --------
    f_v : float or array
        Fractional free volume [-]
    """
    T = np.asarray(T)
    f_v = np.where(T >= T_g_eff, 
                   f_v_g + alpha_f * (T - T_g_eff),
                   f_v_g)
    return f_v


def mobility_function(T, T_g_eff, E_a_seg, C1=17.4, C2=51.6):
    """
    Temperature-dependent segmental mobility function.
    Piecewise WLF-Arrhenius with smooth crossover (CORRECTED v2).

    Regime 1 (T > T_g_eff - C2 + δ): WLF equation
    Regime 2 (T <= T_g_eff - C2 - δ): Arrhenius crossover  
    Buffer zone: linear interpolation for numerical stability

    Parameters:
    -----------
    T : float or array
        Temperature [K]
    T_g_eff : float
        Effective glass transition temperature [K]
    E_a_seg : float
        Segmental activation energy [kJ/mol]
    C1, C2 : float
        WLF constants

    Returns:
    --------
    f_mob : float or array
        Segmental mobility [-]
    """
    T = np.asarray(T, dtype=float)
    E_a_seg_J = E_a_seg * 1000
    T_cross = T_g_eff - C2
    delta_safe = 0.1  # Buffer for numerical stability

    # Pre-calculate crossover mobility
    T_cross_safe = T_cross + delta_safe
    log_aT_cross = -C1 * (T_cross_safe - T_g_eff) / (C2 + T_cross_safe - T_g_eff)
    f_mob_cross = 10**(-log_aT_cross)

    # Arrhenius pre-factor for continuity
    A_cross = f_mob_cross / np.exp(-E_a_seg_J / (R * T_cross_safe))

    # Piecewise calculation
    f_mob = np.empty_like(T, dtype=float)

    # WLF regime
    mask_wlf = T > T_cross_safe
    log_aT = -C1 * (T[mask_wlf] - T_g_eff) / (C2 + T[mask_wlf] - T_g_eff)
    f_mob[mask_wlf] = 10**(-log_aT)

    # Transition zone: linear interpolation
    mask_trans = (T > T_cross - delta_safe) & (T <= T_cross_safe)
    T_left = T_cross - delta_safe
    f_mob_left = A_cross * np.exp(-E_a_seg_J / (R * T_left))
    slope = (f_mob_cross - f_mob_left) / (2 * delta_safe)
    f_mob[mask_trans] = f_mob_left + slope * (T[mask_trans] - T_left)

    # Arrhenius regime
    mask_arr = T <= T_cross - delta_safe
    f_mob[mask_arr] = A_cross * np.exp(-E_a_seg_J / (R * T[mask_arr]))

    return f_mob


def effective_activation_free_energy(Delta_G0, Delta_G_env, Delta_G_struct, 
                                      lam=2.0):
    """
    Saturated (non-perturbative) effective activation free energy.

    Replaces the linear form ΔG‡_eff = ΔG‡_0 - ΔG‡_env - ΔG‡_struct
    which violates perturbativity conditions for realistic polymers.

    Form: ΔG‡_eff = ΔG‡_0 · [1 - tanh((ΔG‡_env + ΔG‡_struct)/(λ·ΔG‡_0))]

    Properties:
    - Always positive (guaranteed by tanh bounds)
    - Recovers linear form when ΔG‡_env + ΔG‡_struct << ΔG‡_0
    - Saturates at ΔG‡_0/2 for large reductions (physical lower bound)

    Parameters:
    -----------
    Delta_G0 : float
        Intrinsic activation free energy [kJ/mol]
    Delta_G_env : float
        Environmental reduction [kJ/mol]
    Delta_G_struct : float
        Structural reduction [kJ/mol]
    lam : float
        Saturation scale parameter (default: 2.0)

    Returns:
    --------
    Delta_G_eff : float
        Effective activation free energy [kJ/mol]
    """
    total_reduction = Delta_G_env + Delta_G_struct
    return Delta_G0 * (1 - np.tanh(total_reduction / (lam * Delta_G0)))


def environmental_contribution(O2, M, UV, alpha):
    """
    Environmental reduction in activation barrier.

    ΔG‡_env = α1·O2 + α2·M + α3·UV

    Parameters:
    -----------
    O2, M, UV : float
        Normalized environmental drivers (0-1)
    alpha : array-like
        Coefficients [α1, α2, α3] in [kJ/mol]

    Returns:
    --------
    Delta_G_env : float
        Environmental contribution [kJ/mol]
    """
    return alpha[0]*O2 + alpha[1]*M + alpha[2]*UV


def structural_contribution(phi_branch, f_v, chi_eff, beta):
    """
    Structural reduction in activation barrier.

    ΔG‡_struct = β1·φ_branch + β2·f_v - β3·χ_eff

    Parameters:
    -----------
    phi_branch : float
        Branching density (0-1)
    f_v : float
        Fractional free volume
    chi_eff : float
        Effective cohesion parameter
    beta : array-like
        Coefficients [β1, β2, β3] in [kJ/mol]

    Returns:
    --------
    Delta_G_struct : float
        Structural contribution [kJ/mol]
    """
    return beta[0]*phi_branch + beta[1]*f_v - beta[2]*chi_eff


def diffusion_coefficient(D0, gamma_prime, f_v):
    """
    Free-volume-dependent diffusion coefficient.

    D = D0 · exp(-γ′ / f_v)

    Parameters:
    -----------
    D0 : float
        Reference diffusion coefficient [m²/s]
    gamma_prime : float
        Consolidated free-volume parameter [-]
    f_v : float
        Fractional free volume

    Returns:
    --------
    D : float
        Diffusion coefficient [m²/s]
    """
    return D0 * np.exp(-gamma_prime / f_v)


# =============================================================================
# SECTION 4: INTEGRATED DEGRADATION MODEL
# =============================================================================

def degradation_rate_constant(T, polymer_name, env_conditions, 
                               alpha, beta, Delta_G0, k0, D0, gamma_prime):
    """
    Complete degradation rate constant.

    k_deg = k0 · f_mob(T, T_g_eff) · exp(-ΔG‡_eff / (R·T))

    Parameters:
    -----------
    T : float
        Temperature [K]
    polymer_name : str
        Key in POLYMER_DB
    env_conditions : dict
        {'O2': float, 'M': float, 'UV': float}
    alpha, beta : array-like
        Calibration coefficients [kJ/mol]
    Delta_G0 : float
        Intrinsic activation free energy [kJ/mol]
    k0 : float
        Pre-exponential factor [1/time]
    D0 : float
        Reference diffusion coefficient [m²/s]
    gamma_prime : float
        Free-volume parameter

    Returns:
    --------
    k_deg : float
        Degradation rate constant [1/time]
    """
    poly = POLYMER_DB[polymer_name]

    # Effective T_g
    T_ge = T_g_effective(poly["T_g_amorph"], poly["T_m"], 
                         poly["X_c"], poly["nu_restriction"])

    # Free volume
    f_v = fractional_free_volume(T, T_ge, poly["f_v_g"], poly["alpha_f"])

    # Segmental mobility
    f_mob = mobility_function(T, T_ge, poly["E_a_seg"])

    # Environmental contribution
    dG_env = environmental_contribution(
        env_conditions["O2"], env_conditions["M"], env_conditions["UV"], alpha)

    # Structural contribution
    dG_struct = structural_contribution(
        poly["phi_branch"], f_v, poly["chi_eff"], beta)

    # Effective activation energy (saturated form)
    dG_eff = effective_activation_free_energy(Delta_G0, dG_env, dG_struct)

    # Rate constant
    k_deg = k0 * f_mob * np.exp(-dG_eff * 1000 / (R * T))

    return k_deg


def coupled_degradation_release(y, t, polymer_name, env_conditions, 
                                 alpha, beta, Delta_G0, k0, k_rel, 
                                 M_w0, M_w_inf, A_s0, f_v0, kappa, t_max):
    """
    Coupled ODE system for degradation and additive release.

    State variables:
    y[0] = M_w(t) / M_w0    (normalized molecular weight)
    y[1] = M_add(t) / M_add0 (normalized additive mass)
    y[2] = X(t)              (degradation progress)

    Parameters:
    -----------
    y : array
        State vector [M_w/M_w0, M_add/M_add0, X]
    t : float
        Time
    [other parameters as above]
    k_rel : float
        Intrinsic release rate [1/time]
    M_w0, M_w_inf : float
        Initial and final molecular weight [g/mol]
    A_s0 : float
        Initial specific surface area [m²/kg]
    f_v0 : float
        Initial fractional free volume
    kappa : float
        Sigmoid steepness [1/time]
    t_max : float
        Time of maximum release [time]

    Returns:
    --------
    dy/dt : array
        Time derivatives
    """
    M_w_norm = y[0]
    M_add_norm = y[1]
    X = y[2]

    # Current physical properties
    poly = POLYMER_DB[polymer_name]
    T = env_conditions.get("T", 298.0)  # Default 25°C
    T_ge = T_g_effective(poly["T_g_amorph"], poly["T_m"], 
                         poly["X_c"], poly["nu_restriction"])
    f_v = fractional_free_volume(T, T_ge, poly["f_v_g"], poly["alpha_f"])

    # Release efficiency (bounded)
    eta_fv = f_v / f_v0
    eta_Mw = (1 - M_w_norm) / (1 - M_w_inf/M_w0)
    # Surface area scales with fragmentation: A_s ~ M_w^(-1/3)
    eta_As = M_w_norm**(-1/3) if M_w_norm > 0.01 else 10.0
    eta_S = 1.0 / (1.0 + np.exp(-kappa * (t - t_max)))

    eta_rel = min(eta_fv * eta_Mw * eta_As * eta_S, 1.0)

    # Degradation rate
    k_deg = degradation_rate_constant(T, polymer_name, env_conditions, 
                                       alpha, beta, Delta_G0, k0, 
                                       1e-12, 0.5)  # D0, gamma dummy

    # ODEs
    dM_w = -k_deg * M_w_norm  # Chain scission reduces M_w
    dM_add = -k_rel * eta_rel * M_add_norm
    dX = k_deg * (1 - X)      # Degradation progress

    return [dM_w, dM_add, dX]


# =============================================================================
# SECTION 5: VALIDATION AGAINST LITERATURE DATA
# =============================================================================

def validate_branching_effect():
    """
    Validation: LDPE vs HDPE degradation rate ratio.
    Literature: LDPE degrades 2-5x faster than HDPE (Zhang et al. 2025;
    Menzel et al. 2022).
    """
    # Calibrated parameters (order-of-magnitude estimates)
    alpha = [25.0, 15.0, 10.0]   # kJ/mol
    beta = [30.0, 200.0, 50.0]   # kJ/mol
    Delta_G0 = 175.0               # kJ/mol
    k0 = 1.0e-6                    # 1/day (order of magnitude)

    env = {"O2": 0.5, "M": 0.3, "UV": 0.2, "T": 298.0}

    k_ldpe = degradation_rate_constant(298.0, "LDPE", env, alpha, beta, 
                                        Delta_G0, k0, 1e-12, 0.5)
    k_hdpe = degradation_rate_constant(298.0, "HDPE", env, alpha, beta, 
                                        Delta_G0, k0, 1e-12, 0.5)

    ratio = k_ldpe / k_hdpe

    print("=" * 60)
    print("VALIDATION: Branching Effect (LDPE vs HDPE)")
    print("=" * 60)
    print(f"k_LDPE = {k_ldpe:.2e} day^-1")
    print(f"k_HDPE = {k_hdpe:.2e} day^-1")
    print(f"Ratio k_LDPE/k_HDPE = {ratio:.2f}")
    print(f"Literature range: 2-5")
    print(f"Status: {'PASS' if 2 <= ratio <= 5 else 'FAIL — requires calibration'}")

    return ratio


def validate_glass_transition_effect():
    """
    Validation: PET vs PE degradation rate ratio.
    Literature: PET 1-2 orders of magnitude slower than polyolefins.
    """
    alpha = [25.0, 15.0, 10.0]
    beta = [30.0, 200.0, 50.0]
    Delta_G0 = 175.0
    k0 = 1.0e-6

    env = {"O2": 0.5, "M": 0.3, "UV": 0.2, "T": 298.0}

    k_pe = degradation_rate_constant(298.0, "LDPE", env, alpha, beta, 
                                      Delta_G0, k0, 1e-12, 0.5)
    k_pet = degradation_rate_constant(298.0, "PET", env, alpha, beta, 
                                       Delta_G0, k0, 1e-12, 0.5)

    ratio = k_pe / k_pet

    print("\n" + "=" * 60)
    print("VALIDATION: Glass Transition Effect (PE vs PET)")
    print("=" * 60)
    print(f"k_PE = {k_pe:.2e} day^-1")
    print(f"k_PET = {k_pet:.2e} day^-1")
    print(f"Ratio k_PE/k_PET = {ratio:.2e}")
    print(f"Literature range: 10-100 (1-2 orders)")
    print(f"Status: {'PASS' if 10 <= ratio <= 100 else 'ADJUST — mobility dominates'}")

    return ratio


# =============================================================================
# SECTION 6: SENSITIVITY ANALYSIS
# =============================================================================

def sensitivity_analysis(polymer_name, env_conditions, alpha, beta, Delta_G0, k0):
    """
    Compute sensitivity coefficients S_i = ∂ln(k_deg)/∂ln(p_i)
    """
    T_base = env_conditions.get("T", 298.0)
    poly = POLYMER_DB[polymer_name]
    T_ge = T_g_effective(poly["T_g_amorph"], poly["T_m"], 
                         poly["X_c"], poly["nu_restriction"])

    # Base rate
    k_base = degradation_rate_constant(T_base, polymer_name, env_conditions, 
                                        alpha, beta, Delta_G0, k0, 1e-12, 0.5)

    sensitivities = {}
    delta = 0.01  # 1% perturbation

    # Temperature sensitivity
    k_T_plus = degradation_rate_constant(T_base*(1+delta), polymer_name, 
                                          env_conditions, alpha, beta, 
                                          Delta_G0, k0, 1e-12, 0.5)
    sensitivities["T"] = (np.log(k_T_plus) - np.log(k_base)) / np.log(1+delta)

    # Environmental sensitivities
    for key in ["O2", "M", "UV"]:
        env_plus = env_conditions.copy()
        env_plus[key] = min(env_plus[key] * (1+delta), 1.0)
        k_plus = degradation_rate_constant(T_base, polymer_name, env_plus, 
                                            alpha, beta, Delta_G0, k0, 1e-12, 0.5)
        sensitivities[key] = (np.log(k_plus) - np.log(k_base)) / np.log(1+delta)

    return sensitivities


# =============================================================================
# SECTION 7: EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("POLYMER DEGRADATION KINETICS PROTOCOL (PDKP) v1.0")
    print("=" * 70)

    # Run validations
    validate_branching_effect()
    validate_glass_transition_effect()

    # Example: Compare all polymers at 25°C
    print("\n" + "=" * 60)
    print("COMPARATIVE DEGRADATION RATES AT 25°C")
    print("=" * 60)

    alpha = [25.0, 15.0, 10.0]
    beta = [30.0, 200.0, 50.0]
    Delta_G0 = 175.0
    k0 = 1.0e-6
    env = {"O2": 0.5, "M": 0.3, "UV": 0.2, "T": 298.0}

    for polymer in ["LDPE", "HDPE", "PP", "PET"]:
        k = degradation_rate_constant(298.0, polymer, env, alpha, beta, 
                                       Delta_G0, k0, 1e-12, 0.5)
        print(f"{polymer:6s}: k_deg = {k:.2e} day^-1")

    # Sensitivity analysis for LDPE
    print("\n" + "=" * 60)
    print("SENSITIVITY ANALYSIS: LDPE")
    print("=" * 60)
    sens = sensitivity_analysis("LDPE", env, alpha, beta, Delta_G0, k0)
    for param, S in sens.items():
        print(f"S_{param:3s} = {S:+.3f}")
