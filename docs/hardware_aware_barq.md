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


## 7. Reproducible 5-seed continuation benchmark

A 10,000-step, 512-point benchmark was run with identical seeds on GitHub
Actions run 36155673396.  The leakage-aware optimizer uses a continuation
strategy: it first converges the compatibility-aware BARQ problem and then
performs 2,500 additional refinement steps with a leakage weight of 0.5.

The common hardware box was

- Omega_max / 2pi = 20 MHz,
- Delta_max / 2pi = 20 MHz,
- phase-slew max / 2pi = 100 MHz,
- transmon anharmonicity / 2pi = -300 MHz.

Common geometric success was defined by closed-curve residual squared below
1e-10 and tangent-area residual squared below 1e-3. Compatibility-aware runs
also required a smooth-polar compatibility barrier below 1e-2. Leakage-aware
runs additionally required maximum transient leakage below 5e-3.

| optimizer | task success | geometric success | mean CFI | mean hardware-box gate time | mean final gate leakage | mean max transient leakage | mean end-to-end optimize time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline BARQ | 5/5 | 5/5 | 4.002e-2 | 888.9 ns | 1.173e-8 | 2.501e-5 | 10.19 s |
| + compatibility | 5/5 | 5/5 | 3.733e-2 | 382.4 ns | 1.717e-9 | 4.714e-5 | 10.44 s |
| + compatibility + leakage continuation | 5/5 | 5/5 | 3.761e-2 | 340.7 ns | 1.085e-12 | 6.463e-5 | 20.51 s |

Relative to baseline, compatibility-aware BARQ reduced mean CFI by 6.73% and
the hardware-box gate-time lower bound by 56.98%, while increasing optimization
time by only 2.49%.

The continuation stage retained 5/5 geometric and compatibility convergence.
Relative to the compatibility-only solution, it reduced the mean gate-time
lower bound by another 10.90% and the final gate leakage by a factor of about
1.58e3, at the cost of a 0.75% increase in CFI and roughly doubling end-to-end
optimization time.  The maximum transient leakage increased modestly but
remained well below the 5e-3 benchmark threshold for every seed.

This result replaces the earlier single-shot leakage-penalty pilot, which lost
geometric convergence for three of five seeds.  It supports continuation as the
default leakage-aware strategy for this prototype, but the five-seed benchmark
is still an algorithm/API validation rather than a statistically final hardware
performance claim.

Raw CSV/JSON results are stored as the GitHub Actions artifact
hardware-aware-barq-benchmark from run 36155673396.
