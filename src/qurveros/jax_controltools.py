"""
Pure-JAX control reconstruction and hardware-gauge utilities.

The original qurveros controltools module intentionally performs interpolation
and post-processing with NumPy/SciPy.  This module provides a differentiable
path for losses that must propagate gradients from hardware-level quantities
back to BARQ/Bezier parameters.
"""

import jax
import jax.numpy as jnp

from qurveros import frametools
from qurveros.settings import settings


def cumulative_trapezoid(y, x):
    """JAX cumulative trapezoid integral along the leading axis."""
    dx = jnp.diff(x)
    if y.ndim == 1:
        inc = 0.5 * (y[:-1] + y[1:]) * dx
        zero = jnp.zeros((1,), dtype=y.dtype)
    else:
        shape = (dx.shape[0],) + (1,) * (y.ndim - 1)
        inc = 0.5 * (y[:-1] + y[1:]) * dx.reshape(shape)
        zero = jnp.zeros((1,) + y.shape[1:], dtype=y.dtype)
    return jnp.concatenate([zero, jnp.cumsum(inc, axis=0)], axis=0)


def _sign_tracker(frenet_dict):
    """Signed-curvature tracker equivalent to phase pi jumps at inflections."""
    B = frenet_dict["frame"][:, 2, :]
    products = jnp.sum(B[1:] * B[:-1], axis=1)
    flips = jnp.where(products < -0.9, -1.0, 1.0)
    return jnp.concatenate([
        jnp.ones((1,), dtype=products.dtype),
        jnp.cumprod(flips),
    ])


@jax.jit
def geometry_control_arrays(frenet_dict):
    """Return dimensionless SCQC arrays on normalized arclength u in [0, 1].

    Returned omega and torsion_hat are T_g Omega and T_g tau respectively.
    """
    x = frenet_dict["x_values"]
    speed = frenet_dict["speed"]

    s = cumulative_trapezoid(speed, x)
    total_length = s[-1]
    u = s / total_length

    sign = _sign_tracker(frenet_dict)
    omega = total_length * sign * frenet_dict["curvature"]
    torsion_hat = total_length * frenet_dict["torsion"]

    return {
        "times": u,
        "omega": omega,
        "torsion": torsion_hat,
        "total_length": total_length,
        "sign_tracker": sign,
    }


def _trap_weights(x):
    """Quadrature weights matching the composite trapezoid rule."""
    dx = jnp.diff(x)
    return jnp.concatenate([
        0.5 * dx[:1],
        0.5 * (dx[:-1] + dx[1:]),
        0.5 * dx[-1:],
    ])


def _tridiagonal_solve(lower, diag, upper, rhs):
    """Differentiable Thomas solve for a tridiagonal linear system."""
    n = diag.shape[0]

    cp0 = upper[0] / diag[0]
    dp0 = rhs[0] / diag[0]

    def fwd(carry, i):
        cp_prev, dp_prev = carry
        denom = diag[i] - lower[i] * cp_prev
        cp = jnp.where(i < n - 1, upper[i] / denom, 0.0)
        dp = (rhs[i] - lower[i] * dp_prev) / denom
        return (cp, dp), (cp, dp)

    (_, _), hist = jax.lax.scan(
        fwd, (cp0, dp0), jnp.arange(1, n)
    )
    cp = jnp.concatenate([jnp.array([cp0]), hist[0]])
    dp = jnp.concatenate([jnp.array([dp0]), hist[1]])

    def bwd(x_next, i):
        x_i = dp[i] - cp[i] * x_next
        return x_i, x_i

    _, reverse_hist = jax.lax.scan(
        bwd, dp[-1], jnp.arange(n - 2, -1, -1)
    )
    return jnp.concatenate([reverse_hist[::-1], dp[-1:]])


@jax.jit
def solve_hardware_gauge(times, torsion_hat, w_detuning=1.0,
                         w_phase=1.0, w_slew=0.0,
                         target_phase_integral=jnp.nan):
    r"""Minimize a quadratic hardware cost at fixed SCQC torsion.

    Let X = T_g dPhi/dt and D = T_g Delta.  The SCQC constraint is

        X - D = T_g tau = torsion_hat.

    The minimized functional is the discrete counterpart of

        int [a D^2 + b X^2 + c (dX/du)^2] du.

    If target_phase_integral is finite, int X du is constrained to that value.
    """
    w = _trap_weights(times)
    du = jnp.diff(times)

    left = jnp.concatenate([
        jnp.zeros((1,), dtype=times.dtype),
        1.0 / du,
    ])
    right = jnp.concatenate([
        1.0 / du,
        jnp.zeros((1,), dtype=times.dtype),
    ])

    lower = -w_slew * left
    upper = -w_slew * right
    diag = (w_detuning + w_phase) * w + w_slew * (left + right)
    rhs = w_detuning * w * torsion_hat

    x_unconstrained = _tridiagonal_solve(lower, diag, upper, rhs)

    def constrained(_):
        z = _tridiagonal_solve(lower, diag, upper, w)
        denom = jnp.dot(w, z)
        lagrange = (
            target_phase_integral - jnp.dot(w, x_unconstrained)
        ) / denom
        return x_unconstrained + lagrange * z

    phase_slew = jax.lax.cond(
        jnp.isfinite(target_phase_integral),
        constrained,
        lambda _: x_unconstrained,
        operand=None,
    )

    delta = phase_slew - torsion_hat
    phi = cumulative_trapezoid(phase_slew, times)

    derivative_cost = jnp.sum(
        w_slew * (jnp.diff(phase_slew)**2) / du
    )
    point_cost = jnp.sum(
        w * (
            w_detuning * delta**2
            + w_phase * phase_slew**2
        )
    )

    return {
        "phase_slew": phase_slew,
        "delta": delta,
        "phi": phi,
        "cost": point_cost + derivative_cost,
    }


def barq_target_phase_integral(frenet_dict):
    """Return the terminal phase integral of the canonical TTC realization.

    This reuses qurveros' own total-torsion-compensation branch choice rather
    than re-detecting frame sign flips.  Preserving this integral therefore
    preserves the same effective BARQ gate (including the singular-frame
    convention) while allowing the time-dependent gauge to change.
    """
    params = frenet_dict["params"]
    if "pgf_params" not in params:
        raise ValueError("BARQ phase target requires pgf_params.")

    angle = jnp.ravel(jnp.asarray(params["pgf_params"]["barq_angle"]))[0]
    total_torsion = frametools.calculate_total_torsion(frenet_dict)
    ttc_detuning = frametools.calculate_ttc_detuning(frenet_dict, angle)
    return total_torsion + ttc_detuning


def hardware_gauge_control(frenet_dict, w_detuning=1.0, w_phase=1.0,
                           w_slew=0.0, preserve_barq_gate=True):
    """Reconstruct a differentiable control using a hardware-optimized gauge."""
    geom = geometry_control_arrays(frenet_dict)

    if preserve_barq_gate:
        target = barq_target_phase_integral(frenet_dict)
    else:
        target = jnp.nan

    gauge = solve_hardware_gauge(
        geom["times"],
        geom["torsion"],
        w_detuning=w_detuning,
        w_phase=w_phase,
        w_slew=w_slew,
        target_phase_integral=target,
    )

    return {
        "times": geom["times"],
        "omega": geom["omega"],
        "torsion": geom["torsion"],
        "phase_slew": gauge["phase_slew"],
        "phi": gauge["phi"],
        "delta": gauge["delta"],
        "gauge_cost": gauge["cost"],
        "total_length": geom["total_length"],
    }


@jax.jit
def minimum_gate_time(omega, delta, phase_slew,
                      omega_max_hw=jnp.inf,
                      delta_max_hw=jnp.inf,
                      phase_slew_max_hw=jnp.inf):
    """Minimum physical gate time imposed by hardware amplitude limits.

    Hardware limits are angular frequencies in rad/s.  The control arrays are
    dimensionless products T_g times the corresponding physical rates.
    """
    t_omega = jnp.max(jnp.abs(omega)) / omega_max_hw
    t_delta = jnp.max(jnp.abs(delta)) / delta_max_hw
    t_phase = jnp.max(jnp.abs(phase_slew)) / phase_slew_max_hw
    return jnp.max(jnp.array([t_omega, t_delta, t_phase]))
