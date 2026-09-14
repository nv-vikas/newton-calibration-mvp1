"""SO-101 peg-insertion task contract.

The pure-Python contract and controller are safe to import without Isaac Lab.
Import :mod:`scene` only after ``AppLauncher`` has started Isaac Sim.
"""

from .contract import (
    PegInsertionMode,
    PegInsertionSceneSpec,
    PegInsertionTaskMetrics,
    local_z_axis_from_quaternion_xyzw,
    required_evidence_channels,
)
from .controller import (
    BilateralContactGripperLatch,
    ControllerCommand,
    ControllerObservation,
    ControllerPhase,
    PegInsertionController,
)
from .robot_adapter import (
    IKSolution,
    SO101_CONTACT_GRIPPER_CENTER_X_M,
    SO101_OPEN_GRIPPER_CENTER_X_M,
    SO101_TASK_TCP_CENTER_X_M,
    SO101_TASK_TCP_OFFSET_GRIPPER_M,
    SO101JointCommandAdapter,
    TCPServoCommand,
    bounded_dls_tcp_step,
    estimate_so101_fingertip_aperture_m,
    interpolate_phase_servo_nominal,
    shift_linear_jacobian_to_point,
    skew_symmetric,
)
from .tcp_setpoint import (
    ProgressiveTCPSetpoint,
    TCP_SETPOINT_SPEED_M_S,
    advance_tcp_setpoint,
    resolve_tcp_setpoint_speeds,
    tcp_setpoint_speed_for_phase,
)

__all__ = [
    "ControllerCommand",
    "BilateralContactGripperLatch",
    "ControllerObservation",
    "ControllerPhase",
    "IKSolution",
    "PegInsertionController",
    "PegInsertionMode",
    "PegInsertionSceneSpec",
    "PegInsertionTaskMetrics",
    "ProgressiveTCPSetpoint",
    "SO101_CONTACT_GRIPPER_CENTER_X_M",
    "SO101_OPEN_GRIPPER_CENTER_X_M",
    "SO101_TASK_TCP_CENTER_X_M",
    "SO101_TASK_TCP_OFFSET_GRIPPER_M",
    "SO101JointCommandAdapter",
    "TCP_SETPOINT_SPEED_M_S",
    "TCPServoCommand",
    "advance_tcp_setpoint",
    "bounded_dls_tcp_step",
    "estimate_so101_fingertip_aperture_m",
    "interpolate_phase_servo_nominal",
    "local_z_axis_from_quaternion_xyzw",
    "required_evidence_channels",
    "resolve_tcp_setpoint_speeds",
    "shift_linear_jacobian_to_point",
    "skew_symmetric",
    "tcp_setpoint_speed_for_phase",
]
