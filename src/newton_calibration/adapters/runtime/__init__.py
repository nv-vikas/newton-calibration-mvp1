from .analytic import AnalyticPDReplayRuntime

__all__ = ["AnalyticPDReplayRuntime"]


def create_runtime(environment):
    if environment.adapter == "analytic":
        return AnalyticPDReplayRuntime(environment)
    if environment.adapter == "isaaclab_newton":
        from .newton import IsaacLabNewtonRuntime

        return IsaacLabNewtonRuntime(environment)
    raise KeyError(f"Unknown runtime adapter: {environment.adapter}")
