from newton_calibration.core.models import ParameterSpec
from newton_calibration.optimizers import DiagonalCMAES


def test_optimizer_converges_on_bounded_quadratic():
    specs = [
        ParameterSpec("x", -5.0, 5.0, 4.0, "", "test", ""),
        ParameterSpec("y", -5.0, 5.0, -4.0, "", "test", ""),
    ]
    optimizer = DiagonalCMAES(specs, population=16, seed=3)
    for _ in range(45):
        candidates = optimizer.ask()
        scores = [(candidate["x"] - 0.75) ** 2 + (candidate["y"] + 1.25) ** 2 for candidate in candidates]
        optimizer.tell(candidates, scores)
    candidate, score = optimizer.best
    assert score < 0.05
    assert abs(candidate["x"] - 0.75) < 0.25
    assert abs(candidate["y"] + 1.25) < 0.25
