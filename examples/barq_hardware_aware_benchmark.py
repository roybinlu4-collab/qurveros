"""Pilot benchmark for hardware-aware BARQ extensions.

This script compares three optimizers on the official X-gate BARQ construction:

1. baseline BARQ,
2. BARQ + state-compatibility endpoint loss,
3. BARQ + state-compatibility + differentiable transmon leakage loss.

The benchmark intentionally uses the same random seeds and optimizer settings.
Defaults are CI-friendly pilot values and can be overridden with environment
variables.
"""

import csv
import json
import os
import time

import jax
import jax.numpy as jnp
import optax
import qutip

from qurveros import barqtools
from qurveros import compatibility
from qurveros import frametools
from qurveros import jax_controltools
from qurveros import losses
from qurveros import transmon
from qurveros.optspacecurve import BarqCurve
from qurveros.qubit_bench import quantumtools
from qurveros.settings import settings


SEEDS = int(os.environ.get("BARQ_BENCHMARK_SEEDS", "5"))
ITERS = int(os.environ.get("BARQ_BENCHMARK_ITERS", "500"))
OPT_POINTS = int(os.environ.get("BARQ_BENCHMARK_OPT_POINTS", "512"))

OMEGA_MAX_HW = 2.0 * jnp.pi * 20e6
DELTA_MAX_HW = 2.0 * jnp.pi * 20e6
PHASE_SLEW_MAX_HW = 2.0 * jnp.pi * 100e6
ANHARMONICITY = -2.0 * jnp.pi * 300e6

GAUGE_W_DETUNING = 4.0
GAUGE_W_PHASE = 1.0
GAUGE_W_SLEW = 0.01

settings.options["OPT_POINTS"] = OPT_POINTS

compat_loss = compatibility.make_population_endpoint_loss(
    theta0=1e-3,
    beta0=jnp.pi / 2.0,
    target_theta=jnp.pi - 1e-3,
    pole_epsilon=1e-4,
)

leakage_loss = transmon.make_transmon_leakage_loss(
    anharmonicity=ANHARMONICITY,
    omega_max_hw=OMEGA_MAX_HW,
    delta_max_hw=DELTA_MAX_HW,
    phase_slew_max_hw=PHASE_SLEW_MAX_HW,
    levels=3,
    n_steps=64,
    max_leakage_weight=10.0,
    w_detuning=GAUGE_W_DETUNING,
    w_phase=GAUGE_W_PHASE,
    w_slew=GAUGE_W_SLEW,
    preserve_barq_gate=True,
)


def make_optimizer(params):
    labels = jax.tree.map(lambda _: True, params)
    labels["pgf_params"]["norm_value"] = False
    return optax.multi_transform(
        {
            True: optax.adam(learning_rate=0.001),
            False: optax.set_to_zero(),
        },
        param_labels=labels,
    )


@jax.jit
def xgate_pgf_mod(pgf_params, input_points):
    del input_points
    pgf_params_mod = pgf_params.copy()
    norm_set = pgf_params_mod['norm_value']
    pgf_params_mod['left_tangent_fix'] = norm_set
    pgf_params_mod['left_tangent_aux'] = norm_set
    pgf_params_mod['left_binormal_fix'] = norm_set
    pgf_params_mod['right_binormal_fix'] = norm_set
    pgf_params_mod['right_tangent_aux'] = norm_set
    pgf_params_mod['right_tangent_fix'] = norm_set
    pgf_params_mod['left_binormal_aux'] = (
        pgf_params_mod['right_binormal_aux']
    )
    return pgf_params_mod


def build_curve(seed, variant):
    adj_target = quantumtools.calculate_adj_rep(qutip.sigmax())
    curve = BarqCurve(
        adj_target=adj_target,
        n_free_points=10,
        pgf_mod=xgate_pgf_mod,
    )
    init_pgf = barqtools.get_default_pgf_params_dict()
    init_pgf['norm_value'] = 0.25
    curve.initialize_parameters(
        seed=4531469 + seed,
        init_pgf_params=init_pgf,
    )

    loss_terms = [
        [losses.tantrix_zero_area_loss, 1.0],
        [losses.max_amp_loss, 1e-2],
    ]
    if variant in ("compatibility", "leakage"):
        loss_terms.append([
            compatibility.smoothstep_polar_compatibility_loss, 1.0
        ])
    if variant == "leakage":
        loss_terms.append([leakage_loss, 5.0])

    curve.prepare_optimization_loss(*loss_terms)
    return curve


def validate(curve, variant, seed, elapsed):
    curve.evaluate_frenet_dict(n_points=2048)
    frenet = curve.get_frenet_dict()

    tan_area = frametools.calculate_tantrix_area(frenet)
    tan_area_sq = float(jnp.sum(tan_area**2))

    total_length = frametools.calculate_total_length(frenet)
    closed_vec = (frenet["curve"][-1] - frenet["curve"][0]) / total_length
    closed_sq = float(jnp.sum(closed_vec**2))

    cfi = float(frametools.calculate_cfi_value(frenet))

    endpoint = float(compat_loss(frenet))
    compatibility_barrier = float(
        compatibility.smoothstep_polar_compatibility_loss(frenet)
    )

    control = jax_controltools.hardware_gauge_control(
        frenet,
        w_detuning=GAUGE_W_DETUNING,
        w_phase=GAUGE_W_PHASE,
        w_slew=GAUGE_W_SLEW,
        preserve_barq_gate=True,
    )
    gate_time = float(jax_controltools.minimum_gate_time(
        control["omega"],
        control["delta"],
        control["phase_slew"],
        omega_max_hw=OMEGA_MAX_HW,
        delta_max_hw=DELTA_MAX_HW,
        phase_slew_max_hw=PHASE_SLEW_MAX_HW,
    ))

    tm = transmon.transmon_metrics_from_frenet(
        frenet,
        anharmonicity=ANHARMONICITY,
        omega_max_hw=OMEGA_MAX_HW,
        delta_max_hw=DELTA_MAX_HW,
        phase_slew_max_hw=PHASE_SLEW_MAX_HW,
        levels=3,
        n_steps=256,
        w_detuning=GAUGE_W_DETUNING,
        w_phase=GAUGE_W_PHASE,
        w_slew=GAUGE_W_SLEW,
        preserve_barq_gate=True,
    )

    leakage = float(tm["leakage"])
    max_leakage = float(tm["max_leakage"])

    geometric_success = bool(
        jnp.isfinite(cfi)
        and tan_area_sq < 1e-3
        and closed_sq < 1e-10
    )
    compatibility_success = compatibility_barrier < 1e-2
    leakage_success = max_leakage < 5e-3

    if variant == "baseline":
        task_success = geometric_success
    elif variant == "compatibility":
        task_success = geometric_success and compatibility_success
    else:
        task_success = (
            geometric_success
            and compatibility_success
            and leakage_success
        )

    return {
        "variant": variant,
        "seed": seed,
        "success": bool(task_success),
        "geometric_success": bool(geometric_success),
        "compatibility_success": bool(compatibility_success),
        "leakage_success": bool(leakage_success),
        "wall_time_s": elapsed,
        "cfi": cfi,
        "gate_time_ns": gate_time * 1e9,
        "leakage": leakage,
        "max_leakage": max_leakage,
        "compatibility_endpoint_loss": endpoint,
        "compatibility_barrier": compatibility_barrier,
        "tantrix_area_sq": tan_area_sq,
        "closed_sq": closed_sq,
        "tg_omega_max": float(jnp.max(jnp.abs(control["omega"]))),
        "tg_delta_max": float(jnp.max(jnp.abs(control["delta"]))),
        "tg_phase_slew_max": float(
            jnp.max(jnp.abs(control["phase_slew"]))
        ),
    }


def main():
    rows = []
    variants = ("baseline", "compatibility", "leakage")

    for seed in range(SEEDS):
        for variant in variants:
            curve = build_curve(seed, variant)
            optimizer = make_optimizer(curve.params)

            start = time.perf_counter()
            curve.optimize(optimizer, max_iter=ITERS)
            jax.block_until_ready(curve.opt_loss(curve.params))
            elapsed = time.perf_counter() - start

            row = validate(curve, variant, seed, elapsed)
            rows.append(row)
            print(json.dumps(row, sort_keys=True))

    with open("barq_hardware_aware_benchmark.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for variant in variants:
        subset = [row for row in rows if row["variant"] == variant]
        good = [row for row in subset if row["success"]]
        denom = max(len(subset), 1)
        summary[variant] = {
            "runs": len(subset),
            "successful_runs": len(good),
            "success_rate": len(good) / denom,
            "geometric_success_rate": sum(
                row["geometric_success"] for row in subset
            ) / denom,
            "mean_wall_time_s": sum(
                row["wall_time_s"] for row in subset
            ) / denom,
            "mean_all_cfi": sum(
                row["cfi"] for row in subset
            ) / denom,
            "mean_all_gate_time_ns": sum(
                row["gate_time_ns"] for row in subset
            ) / denom,
            "mean_all_leakage": sum(
                row["leakage"] for row in subset
            ) / denom,
            "mean_all_max_leakage": sum(
                row["max_leakage"] for row in subset
            ) / denom,
            "mean_success_cfi": (
                sum(row["cfi"] for row in good) / len(good)
                if good else None
            ),
            "mean_success_gate_time_ns": (
                sum(row["gate_time_ns"] for row in good) / len(good)
                if good else None
            ),
            "mean_success_leakage": (
                sum(row["leakage"] for row in good) / len(good)
                if good else None
            ),
            "mean_success_max_leakage": (
                sum(row["max_leakage"] for row in good) / len(good)
                if good else None
            ),
        }

    with open("barq_hardware_aware_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
