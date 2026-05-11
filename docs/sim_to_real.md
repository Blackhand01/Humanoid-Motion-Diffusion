# Sim-to-Real Considerations

Embodied-Motion-Flow currently targets synthetic 12-DOF humanoid joint trajectories. The representation is useful for research iteration, but it is not a complete robot-control interface.

## Synthetic Trajectory Assumptions

- Joint trajectories are represented as time-indexed angular channels in radians.
- Nominal motion is expected to include smooth walking and reaching cycles.
- Anomalous motion is expected to include unstable poses, range-limit violations, abrupt accelerations, temporal jitter, and sensor noise spikes.
- Contact, torque, actuator bandwidth, and full-body dynamics are not part of the initial abstraction.

## 12-DOF Abstraction Limits

- The joint set is abstract and not mapped to a specific humanoid platform.
- Anatomical ranges are configurable research priors and must be replaced with robot-specific limits before deployment.
- The model does not yet encode closed-chain constraints, foot-ground contact consistency, balance, center-of-mass stability, or collision avoidance.

## Generated Motion vs Hardware Commands

Generated trajectories are not directly executable robot commands. A physical robot requires platform-specific retargeting, inverse kinematics, dynamics validation, actuator limits, safety filtering, and low-level control integration.

## Expected Sim-to-Real Gap

- Synthetic motions may be smoother or less constrained than hardware-feasible motions.
- Sensor noise and latency are simplified in the synthetic generator.
- Real robots have backlash, compliance, torque saturation, thermal limits, and controller delay.
- Contact-rich locomotion requires stability checks beyond joint-angle plausibility.

## Safety Considerations

- Never deploy generated trajectories directly on physical hardware.
- Validate trajectories in a physics simulator before hardware tests.
- Enforce robot-specific joint, velocity, acceleration, torque, and collision limits.
- Use conservative speed scaling and emergency-stop procedures for any hardware trial.

## Future Integration Points

- Robot-specific kinematic trees and joint naming.
- Contact-aware dynamics simulation.
- Center-of-mass and zero-moment-point constraints.
- Torque and actuator-bandwidth penalties.
- Retargeting adapters for specific humanoid platforms.
