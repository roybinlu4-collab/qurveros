"""Tests for the pure-JAX hardware-gauge layer."""

import unittest

import jax
import jax.numpy as jnp

from qurveros import jax_controltools


class HardwareGaugeTestCase(unittest.TestCase):

    def test_unconstrained_constant_torsion_solution(self):
        times = jnp.linspace(0.0, 1.0, 129)
        torsion = 2.0 * jnp.ones_like(times)

        control = jax_controltools.solve_hardware_gauge(
            times,
            torsion,
            w_detuning=1.0,
            w_phase=1.0,
            w_slew=0.0,
        )

        self.assertTrue(
            jnp.allclose(control["phase_slew"], 1.0, atol=1e-10)
        )
        self.assertTrue(
            jnp.allclose(control["delta"], -1.0, atol=1e-10)
        )
        self.assertTrue(
            jnp.allclose(
                control["phase_slew"] - control["delta"],
                torsion,
                atol=1e-10,
            )
        )

    def test_terminal_phase_constraint(self):
        times = jnp.linspace(0.0, 1.0, 129)
        torsion = 0.7 + 0.2 * jnp.cos(2*jnp.pi*times)
        target = 0.35

        control = jax_controltools.solve_hardware_gauge(
            times,
            torsion,
            w_detuning=2.0,
            w_phase=1.0,
            w_slew=1e-2,
            target_phase_integral=target,
        )

        integral = jnp.trapezoid(control["phase_slew"], times)
        self.assertTrue(jnp.allclose(integral, target, atol=5e-8))
        self.assertTrue(
            jnp.allclose(
                control["phase_slew"] - control["delta"],
                torsion,
                atol=1e-10,
            )
        )

    def test_gauge_cost_gradient_matches_finite_difference(self):
        times = jnp.linspace(0.0, 1.0, 65)
        base = jnp.sin(2*jnp.pi*times)

        def objective(scale):
            control = jax_controltools.solve_hardware_gauge(
                times,
                scale * base,
                w_detuning=2.0,
                w_phase=1.0,
                w_slew=5e-3,
            )
            return control["cost"]

        x0 = 0.8
        grad_ad = jax.grad(objective)(x0)
        eps = 1e-5
        grad_fd = (objective(x0 + eps) - objective(x0 - eps)) / (2*eps)

        self.assertTrue(jnp.isfinite(grad_ad))
        self.assertTrue(jnp.allclose(grad_ad, grad_fd, rtol=2e-3, atol=2e-5))


if __name__ == "__main__":
    unittest.main()
