"""Tests for differentiable state--SCQC compatibility utilities."""

import unittest

import jax
import jax.numpy as jnp

from qurveros import compatibility


class CompatibilityTestCase(unittest.TestCase):

    def _constant_geometry(self, curvature=1.0, torsion=0.0, n=257):
        x = jnp.linspace(0.0, 0.4, n)
        return {
            "x_values": x,
            "speed": jnp.ones_like(x),
            "curvature": curvature * jnp.ones_like(x),
            "torsion": torsion * jnp.ones_like(x),
        }

    def test_constant_curvature_solution(self):
        frenet = self._constant_geometry()
        theta0 = 0.5 * jnp.pi
        beta0 = 0.5 * jnp.pi

        traj = compatibility.integrate_state_compatibility(
            frenet, theta0, beta0
        )

        expected_theta = theta0 + frenet["x_values"]
        self.assertTrue(
            jnp.allclose(traj["theta"], expected_theta, atol=1e-7)
        )
        self.assertTrue(
            jnp.allclose(traj["beta"], beta0, atol=1e-7)
        )

    def test_endpoint_loss_is_differentiable(self):
        theta0 = 0.5 * jnp.pi
        beta0 = 0.5 * jnp.pi
        target = theta0 + 0.45

        def objective(scale):
            frenet = self._constant_geometry(curvature=scale)
            return compatibility.population_endpoint_loss(
                frenet, theta0, beta0, target
            )

        x0 = 0.9
        grad_ad = jax.grad(objective)(x0)
        eps = 1e-5
        grad_fd = (objective(x0 + eps) - objective(x0 - eps)) / (2*eps)

        self.assertTrue(jnp.isfinite(grad_ad))
        self.assertTrue(jnp.allclose(grad_ad, grad_fd, rtol=2e-3, atol=2e-5))

    def test_smoothstep_margin_detects_obstruction(self):
        n = 257
        x = jnp.linspace(0.0, 1.0, n)
        frenet = {
            "x_values": x,
            "speed": jnp.ones_like(x),
            "curvature": 0.1 * jnp.ones_like(x),
            "torsion": jnp.zeros_like(x),
        }
        loss = compatibility.smoothstep_polar_compatibility_loss(frenet)
        self.assertGreater(float(loss), 0.0)


if __name__ == "__main__":
    unittest.main()
