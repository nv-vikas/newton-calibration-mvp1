from __future__ import annotations

import numpy as np


def _normalized_rmse(predicted: np.ndarray, reference: np.ndarray) -> float:
    rmse = np.sqrt(np.mean(np.square(predicted - reference), axis=0))
    span = np.ptp(reference, axis=0)
    scale = np.maximum(span, 0.05)
    return float(np.mean(rmse / scale))


def _mean_phase_lag(signal_a: np.ndarray, signal_b: np.ndarray, dt: float, max_lag_s: float = 0.25) -> float:
    max_lag = max(1, int(max_lag_s / dt))
    lags: list[float] = []
    for joint in range(signal_a.shape[1]):
        a = signal_a[:, joint] - np.mean(signal_a[:, joint])
        b = signal_b[:, joint] - np.mean(signal_b[:, joint])
        if np.std(a) < 1e-7 or np.std(b) < 1e-7:
            continue
        corr = np.correlate(a, b, mode="full")
        center = len(a) - 1
        window = corr[center - max_lag : center + max_lag + 1]
        lag = int(np.argmax(window)) - max_lag
        lags.append(abs(lag * dt))
    return float(np.mean(lags)) if lags else 0.0


def compare_trajectories(
    simulated_q: np.ndarray,
    simulated_dq: np.ndarray,
    reference_q: np.ndarray,
    reference_dq: np.ndarray,
    command_q: np.ndarray,
    dt: float,
    weights: dict[str, float],
) -> tuple[float, dict[str, float]]:
    length = min(len(simulated_q), len(reference_q), len(command_q))
    simulated_q, simulated_dq = simulated_q[:length], simulated_dq[:length]
    reference_q, reference_dq = reference_q[:length], reference_dq[:length]
    command_q = command_q[:length]
    command_speed = np.linalg.norm(np.gradient(command_q, axis=0), axis=1)
    hold_mask = command_speed <= np.percentile(command_speed, 30)
    hold_error = (
        _normalized_rmse(simulated_q[hold_mask], reference_q[hold_mask])
        if np.count_nonzero(hold_mask) >= 3
        else _normalized_rmse(simulated_q, reference_q)
    )
    metrics = {
        "position_nrmse": _normalized_rmse(simulated_q, reference_q),
        "velocity_nrmse": _normalized_rmse(simulated_dq, reference_dq),
        "phase_error_s": _mean_phase_lag(simulated_q, reference_q, dt),
        "hold_nrmse": hold_error,
        "position_rmse_rad": float(np.sqrt(np.mean(np.square(simulated_q - reference_q)))),
        "position_p95_rad": float(np.percentile(np.abs(simulated_q - reference_q), 95)),
    }
    score = sum(float(weights.get(name, 0.0)) * value for name, value in metrics.items())
    return float(score), metrics
