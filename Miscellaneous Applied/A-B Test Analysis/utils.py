"""
A/B Test Analysis — Utility Functions
Simulate a controlled experiment with a KNOWN true effect, then analyse it with the
standard toolkit: two-proportion z-test, chi-square, confidence interval, and power.
Simulating means we can check the test actually recovers the ground-truth effect.
"""
import numpy as np
from scipy import stats

def simulate_experiment(n_per_group=12000, p_control=0.11, p_treatment=0.128, seed=42):
    """Return per-user conversion arrays for control (A) and treatment (B)."""
    rng = np.random.RandomState(seed)
    A = rng.binomial(1, p_control, n_per_group)
    B = rng.binomial(1, p_treatment, n_per_group)
    return A, B

def two_proportion_ztest(A, B):
    """Pooled two-proportion z-test; returns rates, z, and two-sided p-value."""
    nA, nB = len(A), len(B); cA, cB = A.sum(), B.sum()
    rA, rB = cA/nA, cB/nB
    pool = (cA+cB)/(nA+nB); se = np.sqrt(pool*(1-pool)*(1/nA+1/nB))
    z = (rB-rA)/se; p = 2*(1-stats.norm.cdf(abs(z)))
    return {"rate_A": rA, "rate_B": rB, "z": z, "p_value": p, "abs_uplift": rB-rA,
            "rel_uplift": (rB-rA)/rA}

def uplift_ci(A, B, alpha=0.05):
    """95% confidence interval on the absolute uplift (rate_B - rate_A)."""
    nA, nB = len(A), len(B); rA, rB = A.mean(), B.mean()
    se = np.sqrt(rA*(1-rA)/nA + rB*(1-rB)/nB); zc = stats.norm.ppf(1-alpha/2)
    diff = rB-rA
    return diff-zc*se, diff+zc*se

def chi_square(A, B):
    """Chi-square test of independence on the 2x2 conversion table."""
    cA, cB = A.sum(), B.sum()
    chi2, p, _, _ = stats.chi2_contingency([[cA, len(A)-cA], [cB, len(B)-cB]])
    return chi2, p

def required_sample_size(p_base, mde, alpha=0.05, power=0.8):
    """Approx per-group n to detect an absolute lift `mde` at given alpha/power."""
    z_a = stats.norm.ppf(1-alpha/2); z_b = stats.norm.ppf(power)
    p2 = p_base+mde; pbar = (p_base+p2)/2
    return int(np.ceil(((z_a*np.sqrt(2*pbar*(1-pbar)) + z_b*np.sqrt(p_base*(1-p_base)+p2*(1-p2)))**2)/mde**2))
