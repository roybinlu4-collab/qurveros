"""
Differentiable transmon simulation utilities.

The functions here are intentionally lightweight and JAX-native so leakage
penalties can be differentiated through BARQ control parameters.

The rotating-frame Duffing model is

    H = -Delta n + alpha/2 n(n-1)
        + Omega/2 [exp(-i Phi) a + exp(i Phi) a^dagger].

All dimensional frequencies are angular frequencies in rad/s.
"""

import functools

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg

from qurveros import jax_controltools


def annihilation(levels, dtype=jnp.complex128):
    """Return the truncated harmonic-oscillator annihilation operator."""
    values = jnp.sqrt(jnp.arange(1, levels, dtype=jnp.float64))
    return jnp.diag(values.astype(dtype), k=1)


@functools.partial(jax.jit, static_argnames=("levels",))
def transmon_operators(levels=3):
    """Construct ladder and number operators for a truncated transmon."""
    a = annihilation(levels)
    adag = jnp.conjugate(a.T)
    n = adag @ a
    eye = jnp.eye(levels, dtype=jnp.complex128)
    return a, adag, n, eye


@functools.partial(jax.jit, static_argnames=("levels",))
def transmon_hamiltonian(omega, phi, delta, anharmonicity, levels=3):
    """Rotating-frame Duffing Hamiltonian."""
    a, adag, n, eye = transmon_operators(levels=levels)
    nonlinear = 0.5 * anharmonicity * n @ (n - eye)
    drive = 0.5 * omega * (
        jnp.exp(-1j * phi) * a + jnp.exp(1j * phi) * adag
    )
    return -delta * n + nonlinear + drive


@functools.partial(jax.jit, static_argnames=("levels",))
def propagate_transmon(times, omega, phi, delta, gate_time,
                       anharmonicity, initial_state, levels=3):
    """Piecewise-midpoint propagation of a truncated transmon."""
    dt_u = jnp.diff(times)
    om_mid = 0.5 * (omega[:-1] + omega[1:]) / gate_time
    de_mid = 0.5 * (delta[:-1] + delta[1:]) / gate_time
    ph_mid = jnp.angle(
        jnp.exp(1j * phi[:-1]) + jnp.exp(1j * phi[1:])
    )

    def step(psi, data):
        du, om, ph, de = data
        hamiltonian = transmon_hamiltonian(
            om, ph, de, anharmonicity, levels=levels
        )
        unitary = jsp_linalg.expm(-1j * hamiltonian * gate_time * du)
        psi_next = unitary @ psi
        return psi_next, psi_next

    psi_final, history = jax.lax.scan(
        step,
        initial_state,
        (dt_u, om_mid, ph_mid, de_mid),
    )
    history = jnp.vstack([initial_state[None, :], history])
    return psi_final, history


@functools.partial(jax.jit, static_argnames=("levels",))
def computational_leakage(state, levels=3):
    """Population outside the |0>, |1> computational subspace."""
    del levels
    comp_population = jnp.sum(jnp.abs(state[:2])**2)
    return jnp.maximum(0.0, 1.0 - comp_population)


@functools.partial(jax.jit, static_argnames=("levels",))
def population_transfer_error(state, target_level=1, levels=3):
    """One minus the population in a requested computational target level."""
    del levels
    return 1.0 - jnp.abs(state[target_level])**2


def resample_control(control, n_steps):
    """Resample a differentiable control dictionary to a fixed simulation grid."""
    times_new = jnp.linspace(0.0, 1.0, n_steps)
    times = control["times"]

    omega = jnp.interp(times_new, times, control["omega"])
    delta = jnp.interp(times_new, times, control["delta"])

    phase_complex = jnp.exp(1j * control["phi"])
    phase_real = jnp.interp(times_new, times, jnp.real(phase_complex))
    phase_imag = jnp.interp(times_new, times, jnp.imag(phase_complex))
    phi = jnp.angle(phase_real + 1j * phase_imag)

    return {
        "times": times_new,
        "omega": omega,
        "phi": phi,
        "delta": delta,
    }


def transmon_metrics_from_frenet(
        frenet_dict,
        anharmonicity,
        omega_max_hw,
        delta_max_hw=jnp.inf,
        phase_slew_max_hw=jnp.inf,
        levels=3,
        n_steps=256,
        w_detuning=1.0,
        w_phase=1.0,
        w_slew=0.0,
        preserve_barq_gate=True,
        initial_level=0,
        target_level=1,
):
    """Evaluate leakage and state-transfer error for a BARQ Frenet dictionary."""
    control = jax_controltools.hardware_gauge_control(
        frenet_dict,
        w_detuning=w_detuning,
        w_phase=w_phase,
        w_slew=w_slew,
        preserve_barq_gate=preserve_barq_gate,
    )
    sampled = resample_control(control, n_steps)

    phase_slew = jnp.gradient(sampled["phi"], sampled["times"])
    gate_time = jax_controltools.minimum_gate_time(
        sampled["omega"],
        sampled["delta"],
        phase_slew,
        omega_max_hw=omega_max_hw,
        delta_max_hw=delta_max_hw,
        phase_slew_max_hw=phase_slew_max_hw,
    )

    initial_state = jnp.zeros((levels,), dtype=jnp.complex128)
    initial_state = initial_state.at[initial_level].set(1.0 + 0.0j)

    final_state, history = propagate_transmon(
        sampled["times"],
        sampled["omega"],
        sampled["phi"],
        sampled["delta"],
        gate_time,
        anharmonicity,
        initial_state,
        levels=levels,
    )

    leakage = computational_leakage(final_state, levels=levels)
    transfer_error = population_transfer_error(
        final_state, target_level=target_level, levels=levels
    )
    instantaneous_leakage = 1.0 - jnp.sum(
        jnp.abs(history[:, :2])**2, axis=1
    )

    return {
        "gate_time": gate_time,
        "leakage": leakage,
        "max_leakage": jnp.max(instantaneous_leakage),
        "transfer_error": transfer_error,
        "final_state": final_state,
    }


def make_transmon_leakage_loss(
        anharmonicity,
        omega_max_hw,
        delta_max_hw=jnp.inf,
        phase_slew_max_hw=jnp.inf,
        levels=3,
        n_steps=256,
        max_leakage_weight=0.0,
        w_detuning=1.0,
        w_phase=1.0,
        w_slew=0.0,
        preserve_barq_gate=True,
):
    """Return a differentiable leakage objective for BARQ."""
    def loss(frenet_dict):
        metrics = transmon_metrics_from_frenet(
            frenet_dict,
            anharmonicity=anharmonicity,
            omega_max_hw=omega_max_hw,
            delta_max_hw=delta_max_hw,
            phase_slew_max_hw=phase_slew_max_hw,
            levels=levels,
            n_steps=n_steps,
            w_detuning=w_detuning,
            w_phase=w_phase,
            w_slew=w_slew,
            preserve_barq_gate=preserve_barq_gate,
        )
        return (
            metrics["leakage"]
            + max_leakage_weight * metrics["max_leakage"]
        )

    return jax.jit(loss)


@functools.partial(jax.jit, static_argnames=("levels",))
def propagate_transmon_unitary(times, omega, phi, delta, gate_time,
                               anharmonicity, levels=3):
    """Propagate the full truncated transmon unitary."""
    dt_u = jnp.diff(times)
    om_mid = 0.5 * (omega[:-1] + omega[1:]) / gate_time
    de_mid = 0.5 * (delta[:-1] + delta[1:]) / gate_time
    ph_mid = jnp.angle(
        jnp.exp(1j * phi[:-1]) + jnp.exp(1j * phi[1:])
    )

    identity = jnp.eye(levels, dtype=jnp.complex128)

    def step(unitary_total, data):
        du, om, ph, de = data
        hamiltonian = transmon_hamiltonian(
            om, ph, de, anharmonicity, levels=levels
        )
        unitary_step = jsp_linalg.expm(
            -1j * hamiltonian * gate_time * du
        )
        next_unitary = unitary_step @ unitary_total
        return next_unitary, None

    unitary, _ = jax.lax.scan(
        step,
        identity,
        (dt_u, om_mid, ph_mid, de_mid),
    )
    return unitary


@jax.jit
def average_gate_leakage(unitary):
    """Average final leakage over computational basis inputs |0> and |1>."""
    computational_block = unitary[:2, :2]
    survival = jnp.sum(jnp.abs(computational_block)**2) / 2.0
    return jnp.maximum(0.0, 1.0 - survival)


def gate_leakage_metrics_from_frenet(
        frenet_dict,
        anharmonicity,
        omega_max_hw,
        delta_max_hw=jnp.inf,
        phase_slew_max_hw=jnp.inf,
        levels=3,
        n_steps=256,
        w_detuning=1.0,
        w_phase=1.0,
        w_slew=0.0,
        preserve_barq_gate=True,
):
    """Evaluate gate-level leakage for a BARQ Frenet dictionary."""
    control = jax_controltools.hardware_gauge_control(
        frenet_dict,
        w_detuning=w_detuning,
        w_phase=w_phase,
        w_slew=w_slew,
        preserve_barq_gate=preserve_barq_gate,
    )
    sampled = resample_control(control, n_steps)

    phase_slew = jnp.gradient(sampled["phi"], sampled["times"])
    gate_time = jax_controltools.minimum_gate_time(
        sampled["omega"],
        sampled["delta"],
        phase_slew,
        omega_max_hw=omega_max_hw,
        delta_max_hw=delta_max_hw,
        phase_slew_max_hw=phase_slew_max_hw,
    )

    unitary = propagate_transmon_unitary(
        sampled["times"],
        sampled["omega"],
        sampled["phi"],
        sampled["delta"],
        gate_time,
        anharmonicity,
        levels=levels,
    )

    return {
        "gate_time": gate_time,
        "leakage": average_gate_leakage(unitary),
        "unitary": unitary,
    }


def make_transmon_gate_leakage_loss(
        anharmonicity,
        omega_max_hw,
        delta_max_hw=jnp.inf,
        phase_slew_max_hw=jnp.inf,
        levels=3,
        n_steps=256,
        w_detuning=1.0,
        w_phase=1.0,
        w_slew=0.0,
        preserve_barq_gate=True,
):
    """Return an average gate-leakage objective for BARQ optimization."""
    def loss(frenet_dict):
        metrics = gate_leakage_metrics_from_frenet(
            frenet_dict,
            anharmonicity=anharmonicity,
            omega_max_hw=omega_max_hw,
            delta_max_hw=delta_max_hw,
            phase_slew_max_hw=phase_slew_max_hw,
            levels=levels,
            n_steps=n_steps,
            w_detuning=w_detuning,
            w_phase=w_phase,
            w_slew=w_slew,
            preserve_barq_gate=preserve_barq_gate,
        )
        return metrics["leakage"]

    return jax.jit(loss)
