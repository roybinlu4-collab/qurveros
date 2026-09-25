"""
Tests for state compatibility, hardware gauge, and differentiable leakage.
"""

import unittest

import jax
import jax.numpy as jnp

from qurveros import compatibility, jax_controltools, transmon


class CompatibilityTestCase(unittest.TestCase):

    def test_constant_curvature_solution(self):
        x = jnp.linspace(0.0, 0.5, 257)
        frenet_dict = {
            "x_values": x,
            "speed": jnp.ones_like(x),
            "curvature": jnp.ones_like(x),
            "torsion": jnp.zeros_like(x),
        }

        theta0 = 0.7
        beta0 = jnp.pi / 2.0
        traj = compatibility.integrate_state_compatibility(
            frenet_dict, theta0, beta0
        )

        self.assertTrue(
            jnp.allclose(traj["theta"], theta0 + x, atol=2e-7)
        )
        self.assertTrue(
            jnp.allclose(traj["beta"], beta0, atol=2e-7)
        )

    def test_endpoint_loss_is_differentiable(self):
        x = jnp.linspace(0.0, 0.25, 129)

        def objective(scale):
            frenet_dict = {
                "x_values": x,
                "speed": jnp.ones_like(x),
                "curvature": scale * jnp.ones_like(x),
                "torsion": jnp.zeros_like(x),
            }
            return compatibility.population_endpoint_loss(
                frenet_dict,
                theta0=0.6,
                beta0=jnp.pi / 2.0,
                target_theta=1.0,
            )

        grad = jax.grad(objective)(jnp.array(1.0))
        self.assertTrue(jnp.isfinite(grad))


class HardwareGaugeTestCase(unittest.TestCase):

    def test_torsion_is_gauge_invariant(self):
        times = jnp.linspace(0.0, 1.0, 257)
        torsion = 0.4 + 0.3 * jnp.sin(2.0 * jnp.pi * times)

        gauge = jax_controltools.solve_hardware_gauge(
            times,
            torsion,
            w_detuning=3.0,
            w_phase=1.0,
            w_slew=0.02,
        )

        recovered = gauge["phase_slew"] - gauge["delta"]
        self.assertTrue(jnp.allclose(recovered, torsion, atol=1e-11))

    def test_zero_slew_has_analytic_solution(self):
        times = jnp.linspace(0.0, 1.0, 129)
        torsion = jnp.cos(2.0 * jnp.pi * times)

        gauge = jax_controltools.solve_hardware_gauge(
            times,
            torsion,
            w_detuning=3.0,
            w_phase=2.0,
            w_slew=0.0,
        )

        expected = 3.0 / 5.0 * torsion
        self.assertTrue(
            jnp.allclose(gauge["phase_slew"], expected, atol=1e-10)
        )


class TransmonTestCase(unittest.TestCase):

    def test_zero_drive_has_zero_leakage(self):
        times = jnp.linspace(0.0, 1.0, 65)
        zeros = jnp.zeros_like(times)
        initial = jnp.array([1.0, 0.0, 0.0], dtype=jnp.complex128)

        final, _ = transmon.propagate_transmon(
            times,
            zeros,
            zeros,
            zeros,
            gate_time=40e-9,
            anharmonicity=-2.0*jnp.pi*300e6,
            initial_state=initial,
            levels=3,
        )
        leakage = transmon.computational_leakage(final, levels=3)
        self.assertTrue(jnp.isclose(leakage, 0.0, atol=1e-12))

    def test_leakage_gradient_matches_finite_difference(self):
        times = jnp.linspace(0.0, 1.0, 65)
        phi = jnp.zeros_like(times)
        delta = jnp.zeros_like(times)
        initial = jnp.array([1.0, 0.0, 0.0], dtype=jnp.complex128)
        gate_time = 40e-9
        alpha = -2.0*jnp.pi*300e6

        def loss(amplitude):
            omega = amplitude * jnp.sin(jnp.pi * times)**2
            final, _ = transmon.propagate_transmon(
                times,
                omega,
                phi,
                delta,
                gate_time=gate_time,
                anharmonicity=alpha,
                initial_state=initial,
                levels=3,
            )
            return transmon.computational_leakage(final, levels=3)

        amp = jnp.array(7.0)
        autodiff = jax.grad(loss)(amp)
        eps = 1e-3
        finite_diff = (loss(amp + eps) - loss(amp - eps)) / (2.0*eps)

        self.assertTrue(jnp.isfinite(autodiff))
        self.assertTrue(
            jnp.allclose(autodiff, finite_diff, rtol=5e-3, atol=5e-7)
        )


if __name__ == "__main__":
    unittest.main()
