"""Tests for the differentiable transmon leakage model."""

import unittest

import jax
import jax.numpy as jnp

from qurveros import transmon


class TransmonTestCase(unittest.TestCase):

    def test_zero_drive_has_zero_leakage(self):
        times = jnp.linspace(0.0, 1.0, 33)
        omega = jnp.zeros_like(times)
        phi = jnp.zeros_like(times)
        delta = jnp.zeros_like(times)
        state0 = jnp.array([1.0 + 0.0j, 0.0j, 0.0j])

        final_state, _ = transmon.propagate_transmon(
            times,
            omega,
            phi,
            delta,
            gate_time=20e-9,
            anharmonicity=-2*jnp.pi*300e6,
            initial_state=state0,
            levels=3,
        )

        leakage = transmon.computational_leakage(final_state, levels=3)
        self.assertTrue(jnp.allclose(leakage, 0.0, atol=1e-12))


    def test_gate_level_zero_drive_has_zero_leakage(self):
        times = jnp.linspace(0.0, 1.0, 33)
        zeros = jnp.zeros_like(times)

        unitary = transmon.propagate_transmon_unitary(
            times,
            zeros,
            zeros,
            zeros,
            gate_time=20e-9,
            anharmonicity=-2*jnp.pi*300e6,
            levels=3,
        )

        leakage = transmon.average_gate_leakage(unitary)
        self.assertTrue(jnp.allclose(leakage, 0.0, atol=1e-12))

    def test_two_level_pi_pulse(self):
        times = jnp.linspace(0.0, 1.0, 129)
        omega = jnp.pi * jnp.ones_like(times)
        phi = jnp.zeros_like(times)
        delta = jnp.zeros_like(times)
        state0 = jnp.array([1.0 + 0.0j, 0.0j])

        final_state, _ = transmon.propagate_transmon(
            times,
            omega,
            phi,
            delta,
            gate_time=1.0,
            anharmonicity=0.0,
            initial_state=state0,
            levels=2,
        )

        self.assertTrue(
            jnp.allclose(jnp.abs(final_state[1])**2, 1.0, atol=1e-9)
        )

    def test_leakage_is_differentiable(self):
        times = jnp.linspace(0.0, 1.0, 65)
        phi = jnp.zeros_like(times)
        delta = jnp.zeros_like(times)
        state0 = jnp.array([1.0 + 0.0j, 0.0j, 0.0j])
        gate_time = 20e-9
        anh = -2*jnp.pi*300e6

        def objective(scale):
            omega = scale * jnp.pi * jnp.ones_like(times)
            final_state, _ = transmon.propagate_transmon(
                times,
                omega,
                phi,
                delta,
                gate_time=gate_time,
                anharmonicity=anh,
                initial_state=state0,
                levels=3,
            )
            return transmon.computational_leakage(final_state, levels=3)

        value, grad = jax.value_and_grad(objective)(1.0)
        self.assertTrue(jnp.isfinite(value))
        self.assertTrue(jnp.isfinite(grad))


if __name__ == "__main__":
    unittest.main()
