#!/usr/bin/env python3
"""Research-grade validation tests."""

import sys
sys.path.insert(0, 'src')

import numpy as np
from physics import (
    compute_lagrange_points, 
    validate_lagrange_points,
    validate_jacobi_conservation,
    propagate,
    EARTH_MOON,
    SUN_EARTH
)

print("=" * 60)
print("CISLUNAR SSA - RESEARCH GRADE VALIDATION")
print("=" * 60)

print("\n[1] Environment Validation")
print(f"  Earth-Moon mass ratio: {EARTH_MOON.mass_ratio:.12f}")
print(f"  Expected:              0.012150585609624")
print(f"  Match: {abs(EARTH_MOON.mass_ratio - 0.012150585609624) < 1e-15}")

print("\n[2] Lagrange Point Validation against NASA/JPL")
L = compute_lagrange_points()
validation = validate_lagrange_points(L, tolerance=1e-6)

all_passed = True
for name, result in validation.items():
    status = "PASS" if result['passed'] else "FAIL"
    if not result['passed']:
        all_passed = False
    print(f"  {name}: error = {result['error']:.2e} [{status}]")

print(f"\n  Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

print("\n[3] Jacobi Integral Conservation (Long-term)")
state0 = [L['L1'][0] + 0.01, 0.0, 0.0, 0.0, 0.1, 0.0]
t_eval = np.linspace(0, 50, 5000)  # Long integration
sol = propagate(state0, (0, 50), t_eval)

jacobi_test = validate_jacobi_conservation(state0, sol.y[:, -1], tolerance=1e-7)
print(f"  Integration time: 50 non-dimensional units (~7 months)")
print(f"  Initial C: {jacobi_test['initial_C']:.10f}")
print(f"  Final C:   {jacobi_test['final_C']:.10f}")
print(f"  Error:     {jacobi_test['error']:.2e}")
print(f"  Status:    {'PASS' if jacobi_test['passed'] else 'FAIL'}")

print("\n[4] Physical Scale Conversion")
# Convert non-dimensional to physical units
L_EM = 384400.0  # km
T_EM = 4.3425 * 24 * 3600  # seconds (~4.34 days, half Moon period)

L1_physical = L['L1'][0] * L_EM
print(f"  L1 distance from Earth: {L1_physical:.1f} km")
print(f"  Expected: ~321,000 km")
print(f"  Match: {abs(L1_physical - 321000) < 5000}")

print("\n" + "=" * 60)
print("VALIDATION COMPLETE")
print("=" * 60)
