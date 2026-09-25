# Hardware-aware BARQ extensions

This draft adds an opt-in JAX path for connecting BARQ/SCQC geometry to
state-trajectory feasibility and weakly anharmonic hardware.

## 1. State--SCQC compatibility

For a Bloch trajectory with polar angle theta and relative angle
beta = Phi - alpha, the state and SCQC curve must satisfy

dtheta/dt = kappa sin(beta),
dbeta/dt = tau + kappa cos(beta) cot(theta).

The compatibility module integrates these equations directly from a Frenet
dictionary using a JAX RK4 scan. It also exposes a differentiable projective
endpoint loss and a local feasibility barrier for a smooth polar transfer.

The local barrier implements the necessary condition

Tg * kappa(u) >= |dtheta/du|.

It is useful as an inexpensive preconditioner; it is not a replacement for the
full coupled compatibility equations.

## 2. Hardware gauge at fixed robust geometry

SCQC fixes

X - D = Tg * tau,

with X = Tg * dPhi/dt and D = Tg * Delta.

The new JAX control path minimizes a discrete version of

Jg = integral [a D^2 + b X^2 + c (dX/du)^2] du

without changing the curve curvature or torsion. An optional phase-integral
constraint preserves the complete BARQ gate. The implementation solves the
resulting tridiagonal Euler--Lagrange system differentiably.

Example:

    from qurveros import jax_controltools

    control = jax_controltools.hardware_gauge_control(
        frenet_dict,
        w_detuning=4.0,
        w_phase=1.0,
        w_slew=0.01,
        preserve_barq_gate=True,
    )

## 3. Hardware resource box

Dimensionless pulse resources are converted to a physical lower bound

Tmin = max(
    ||Tg Omega||_inf / Omega_max_hw,
    ||Tg Delta||_inf / Delta_max_hw,
    ||Tg dPhi/dt||_inf / phase_slew_max_hw
).

This prevents an optimizer from appearing faster by moving cost from one
control channel to another.

## 4. Differentiable transmon leakage

The transmon module uses the truncated rotating-frame Duffing model

H(t) = -Delta(t) n + alpha/2 n(n-1)
       + Omega(t)/2 [exp(-i Phi(t)) a + exp(i Phi(t)) a^dagger].

The midpoint propagator uses jax.lax.scan and jax.scipy.linalg.expm, allowing
leakage gradients to flow back to BARQ parameters. Both final leakage and
maximum transient leakage are available.

Example:

    from qurveros import transmon

    leakage_loss = transmon.make_transmon_leakage_loss(
        anharmonicity=-2*jnp.pi*300e6,
        omega_max_hw=2*jnp.pi*20e6,
        delta_max_hw=2*jnp.pi*20e6,
        phase_slew_max_hw=2*jnp.pi*100e6,
        levels=3,
        max_leakage_weight=10.0,
    )

## 5. Benchmark

examples/barq_hardware_aware_benchmark.py uses the same X-gate PGF
construction, norm_value=0.25, Adam learning rate, and fixed-norm parameter
mask as examples/Xgate_design_barq.ipynb.

It compares identical seeds for

1. baseline BARQ,
2. BARQ plus a state-feasibility compatibility barrier,
3. BARQ plus compatibility and transmon leakage.

The output records common geometric success, variant-specific success, wall
time, CFI, hardware-box gate time, final/transient leakage, tangent-area
residual, closed-curve residual, and dimensionless peak resources.

This benchmark is intended to diagnose tradeoffs rather than imply that every
metric must improve simultaneously. In particular, suppressing leakage can
trade against gate time or other geometric costs.

## 6. Backward compatibility

The public default remains unchanged: BarqCurve.evaluate_control_dict() still
means TTC control. An explicit mode can now be supplied, and all new
hardware-aware behavior is opt-in.


## 7. Reproducible 5-seed pilot result

A 10,000-step, 512-point pilot was run with identical seeds on GitHub Actions
(run 36122394505). The common hardware box used

- Omega_max / 2pi = 20 MHz,
- Delta_max / 2pi = 20 MHz,
- phase-slew max / 2pi = 100 MHz,
- transmon anharmonicity / 2pi = -300 MHz.

Common geometric success was defined by closed-curve residual squared below
1e-10 and tangent-area residual squared below 1e-3. Compatibility-aware runs
also required a smooth-polar compatibility barrier below 1e-2. Leakage-aware
runs additionally required maximum transient leakage below 5e-3.

| optimizer | task success | geometric success | mean CFI | mean hardware-box gate time | mean final leakage | mean max transient leakage | mean optimize time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline BARQ | 5/5 | 5/5 | 4.004e-2 | 889.0 ns | 1.786e-8 | 2.495e-5 | 13.28 s |
| + compatibility | 5/5 | 5/5 | 3.735e-2 | 382.3 ns | 2.366e-9 | 4.698e-5 | 13.94 s |
| + compatibility + leakage | 2/5 | 2/5 | 2.745e-2 (all runs) | 637.5 ns (all runs) | 2.111e-6 (all runs) | 5.553e-4 (all runs) | 44.20 s |

For the two leakage-aware runs that met all success criteria, the mean CFI was
2.826e-2, the mean hardware-box gate time was 489.0 ns, and the mean final
leakage was 5.327e-11.

The strongest pilot observation is not a universal leakage advantage. It is
that the compatibility barrier reduced its mean value from 3.975e-1 to
2.369e-4 while retaining 5/5 geometric convergence. Under the selected common
hardware box this coincided with a 57.0% lower mean hardware-limited gate time
and a 6.73% lower mean CFI than baseline, with only about 5% additional
optimization time.

The leakage-aware objective is not yet tuned: it reduced CFI in the converged
subset but only 2/5 seeds met the common geometric constraint. This is a useful
negative result and indicates that leakage should be introduced by continuation
or an augmented-Lagrangian/Pareto strategy rather than by a single large
weighted penalty.
