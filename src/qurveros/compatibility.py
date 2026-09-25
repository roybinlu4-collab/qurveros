"""
State--SCQC compatibility utilities.

This module implements the gauge-invariant compatibility equations between a
Bloch-state trajectory and the curvature/torsion data of an SCQC space curve.

For a state polar angle theta and relative phase beta = Phi - alpha,

    d theta / dt = kappa sin(beta)
    d beta  / dt = tau + kappa cos(beta) cot(theta).

For a non-unit-speed curve parametrized by x, dt = ds = speed * dx, so the
right-hand sides are multiplied by the curve speed.

The implementation is pure JAX and can therefore be used inside BARQ losses.
"""

import functools

import jax
import jax.numpy as jnp


def _safe_cot(theta, pole_epsilon):
    """Numerically regularized cotangent for spherical-coordinate poles."""
    s = jnp.sin(theta)
    sign = jnp.where(s >= 0.0, 1.0, -1.0)
    s_safe = jnp.where(jnp.abs(s) < pole_epsilon,
                       sign * pole_epsilon, s)
    return jnp.cos(theta) / s_safe


@functools.partial(jax.jit, static_argnames=("pole_epsilon",))
def compatibility_rhs(theta, beta, speed, curvature, torsion,
                      pole_epsilon=1e-7):
    """Compatibility equations written with respect to the curve parameter."""
    dtheta_dx = speed * curvature * jnp.sin(beta)
    dbeta_dx = speed * (
        torsion
        + curvature * jnp.cos(beta) * _safe_cot(theta, pole_epsilon)
    )
    return jnp.array([dtheta_dx, dbeta_dx])


@functools.partial(jax.jit, static_argnames=("pole_epsilon",))
def integrate_state_compatibility(frenet_dict, theta0, beta0,
                                  pole_epsilon=1e-7):
    """Integrate the state--SCQC compatibility equations with RK4.

    Args:
        frenet_dict: A qurveros Frenet dictionary.
        theta0: Initial Bloch polar angle.
        beta0: Initial relative phase beta = Phi - alpha.
        pole_epsilon: Regularization used only in cot(theta).

    Returns:
        A dictionary with arrays theta and beta sampled at the same
        curve-parameter values as frenet_dict.
    """
    x = frenet_dict["x_values"]
    speed = frenet_dict["speed"]
    curvature = frenet_dict["curvature"]
    torsion = frenet_dict["torsion"]

    y0 = jnp.array([theta0, beta0], dtype=x.dtype)

    x0 = x[:-1]
    x1 = x[1:]
    s0 = speed[:-1]
    s1 = speed[1:]
    k0 = curvature[:-1]
    k1 = curvature[1:]
    t0 = torsion[:-1]
    t1 = torsion[1:]

    def rhs(y, s, k, tau):
        return compatibility_rhs(
            y[0], y[1], s, k, tau, pole_epsilon=pole_epsilon
        )

    def step(y, data):
        xa, xb, sa, sb, ka, kb, taua, taub = data
        h = xb - xa
        sm = 0.5 * (sa + sb)
        km = 0.5 * (ka + kb)
        taum = 0.5 * (taua + taub)

        k_1 = rhs(y, sa, ka, taua)
        k_2 = rhs(y + 0.5 * h * k_1, sm, km, taum)
        k_3 = rhs(y + 0.5 * h * k_2, sm, km, taum)
        k_4 = rhs(y + h * k_3, sb, kb, taub)

        y_next = y + (h / 6.0) * (k_1 + 2.0*k_2 + 2.0*k_3 + k_4)
        return y_next, y_next

    data = (x0, x1, s0, s1, k0, k1, t0, t1)
    _, ys = jax.lax.scan(step, y0, data)
    ys = jnp.vstack([y0, ys])

    return {
        "theta": ys[:, 0],
        "beta": ys[:, 1],
    }


@functools.partial(jax.jit, static_argnames=("pole_epsilon",))
def population_endpoint_loss(frenet_dict, theta0, beta0, target_theta,
                             pole_epsilon=1e-7):
    """Projective endpoint loss for a state-transfer task.

    The loss compares the Bloch z coordinate, so azimuth is intentionally
    ignored. This is appropriate for north/south-pole population transfer and
    for task-aware objectives where a terminal phase is not physically fixed.
    """
    traj = integrate_state_compatibility(
        frenet_dict, theta0, beta0, pole_epsilon=pole_epsilon
    )
    z_final = jnp.cos(traj["theta"][-1])
    z_target = jnp.cos(target_theta)
    return (z_final - z_target)**2


def make_population_endpoint_loss(theta0, beta0, target_theta,
                                  pole_epsilon=1e-7):
    """Factory returning a loss(frenet_dict) compatible with BARQ."""
    @jax.jit
    def loss(frenet_dict):
        return population_endpoint_loss(
            frenet_dict,
            theta0,
            beta0,
            target_theta,
            pole_epsilon=pole_epsilon,
        )
    return loss


@jax.jit
def prescribed_theta_margin(frenet_dict, theta):
    """Return kappa - abs(d theta / ds) for a prescribed polar trajectory."""
    x = frenet_dict["x_values"]
    speed = frenet_dict["speed"]
    dtheta_dx = jnp.gradient(theta, x)
    dtheta_ds = dtheta_dx / speed
    return frenet_dict["curvature"] - jnp.abs(dtheta_ds)


@jax.jit
def smoothstep_polar_compatibility_loss(frenet_dict, theta_span=jnp.pi):
    """Penalize violation of kappa >= abs(d theta/ds) for a smooth transfer.

    The prescribed path uses theta(u) = theta_span * (3 u**2 - 2 u**3),
    where u is normalized arclength.  The loss is zero when the local
    state-speed necessary condition is satisfied everywhere.
    """
    x = frenet_dict['x_values']
    speed = frenet_dict['speed']
    dx = jnp.diff(x)
    increments = 0.5 * (speed[:-1] + speed[1:]) * dx
    s = jnp.concatenate([jnp.zeros((1,), dtype=x.dtype),
                         jnp.cumsum(increments)])
    total_length = s[-1]
    u = s / total_length
    required = jnp.abs(theta_span) * 6.0 * u * (1.0 - u)
    available = total_length * frenet_dict['curvature']
    violation = jax.nn.relu(required - available)
    return jnp.mean(violation**2)
