import math
import logging
import numpy as np
from enum import Enum

logger = logging.getLogger('gym_agx.rl')


class EndEffectorConstraint:
    class Dof(Enum):
        X_TRANSLATION = 0,
        Y_TRANSLATION = 1,
        Z_TRANSLATION = 2,
        X_ROTATION = 3,
        Y_ROTATION = 4,
        Z_ROTATION = 5,
        X_COMPLIANCE = 6,
        Y_COMPLIANCE = 7,
        Z_COMPLIANCE = 8,

    def __init__(self, end_effector_dof, compute_forces_enabled, velocity_control, compliance_control, velocity_index,
                 compliance_index):
        """EndEffectorConstraint class, defining important parameters of individual constraints.
        :param EndEffectorConstraint.Dof end_effector_dof: degree of freedom of end-effector controlled
        :param bool compute_forces_enabled: force and torque can be measured (should be consistent with simulation)
        :param bool velocity_control: is velocity controlled
        :param bool compliance_control: is compliance controlled
        :param int velocity_index: index of action vector which controls velocity of this constraint's motor
        :param int compliance_index: index of action vector which controls compliance of this constraint's motor
        """
        self.end_effector_dof = end_effector_dof
        self.compute_forces_enabled = compute_forces_enabled
        self.velocity_control = velocity_control
        self.compliance_control = compliance_control
        self.velocity_index = velocity_index
        self.compliance_index = compliance_index

    @property
    def is_active(self):
        return True if self.velocity_control or self.compliance_control else False


class EndEffector:
    action_indices = dict()

    def __init__(self, name, controllable, observable, max_velocity=1, max_angular_velocity=1, max_acceleration=1,
                 max_angular_acceleration=1, min_compliance=0, max_compliance=1e6):
        """EndEffector class which keeps track of end-effector constraints and action indices.
        :param str name: Name of the end-effector, should match name of rigid body in simulation.
        :param bool controllable: Determines if the end-effector is controllable.
        :param bool observable: Determines if the end-effector is observable.
        :param float max_velocity: Maximum velocity sent to simulation.
        :param float max_angular_velocity: Maximum angular velocity sent to simulation.
        :param float max_acceleration: Maximum acceleration allowed. Affects velocity sent to simulation.
        :param float max_angular_acceleration: Maximum angular acceleration allowed. Affects angular velocity sent to
        simulation.
        :param float min_compliance: Minimum compliance of the end-effector grip.
        :param float max_compliance: Maximum compliance of the end-effector grip.
        """
        self.name = name
        self.controllable = controllable
        self.observable = observable
        self.max_velocity = max_velocity
        self.max_angular_velocity = max_angular_velocity
        self.max_acceleration = max_acceleration
        self.max_angular_acceleration = max_angular_acceleration
        self.min_compliance = min_compliance
        self.max_compliance = max_compliance
        self.constraints = {}

    def add_constraint(self, name, end_effector_dof, compute_forces_enabled=False, velocity_control=False,
                       compliance_control=False):
        """Add constraints which make up the end-effector.
        :param str name: Name of the constraint. Should be consistent with name of constraint in simulation.
        :param EndEffectorConstraint.Dof end_effector_dof: DoF of end-effector that this constraint controls
        :param bool compute_forces_enabled: Force and torque can be measured (should be consistent with simulation)
        :param bool velocity_control: Is velocity controlled
        :param bool compliance_control: Is compliance controlled
        """
        velocity_index = None
        compliance_index = None
        if velocity_control:
            velocity_constraint_name = name + '_velocity'
            if velocity_constraint_name not in self.action_indices:
                self.action_indices[velocity_constraint_name] = len(self.action_indices)
            velocity_index = EndEffector.action_indices[velocity_constraint_name]
        if compliance_control:
            compliance_constraint_name = name + '_compliance'
            if compliance_constraint_name not in self.action_indices:
                self.action_indices[compliance_constraint_name] = len(self.action_indices)
            compliance_index = self.action_indices[compliance_constraint_name]

        end_effector_constraint = EndEffectorConstraint(end_effector_dof, compute_forces_enabled, velocity_control,
                                                        compliance_control, velocity_index, compliance_index)
        self.constraints.update({name: end_effector_constraint})

    def apply_control(self, sim, action, dt):
        """Apply control to simulation.
        :param agxSDK.Simulation sim: AGX simulation object.
        :param np.ndarray action: Action from Gym interface.
        :param float dt: Action time-step, needed to compute velocity and acceleration.
        :return: Applied actions
        """
        control_actions = []
        if self.controllable:
            for key, constraint in self.constraints.items():
                joint = sim.getConstraint1DOF(key)
                motor = joint.getMotor1D()
                if constraint.velocity_control:
                    current_velocity, linear = self.get_velocity(sim, constraint.end_effector_dof)
                    velocity = self.rescale_velocity(action[constraint.velocity_index], current_velocity, dt, linear)
                    motor.setSpeed(np.float64(velocity))
                    control_actions.append(velocity)
                if constraint.compliance_control:
                    motor_param = motor.getRegularizationParameters()
                    compliance = self.rescale_compliance(action[constraint.compliance_index])
                    motor_param.setCompliance(np.float64(compliance))
                    control_actions.append(compliance)
        else:
            logger.debug("Received apply_control command for uncontrollable end-effector.")

        return control_actions

    def get_velocity(self, sim, constraint_dof):
        """Get current velocity of end_effector.
        :param agxSDK.Simulation sim: AGX simulation object.
        :param EndEffectorConstraint.Dof constraint_dof: Degree of freedom to read velocity from.
        :return: End-effector velocity and boolean indicating if it is linear or angular.
        """
        end_effector = sim.getRigidBody(self.name)
        if constraint_dof == EndEffectorConstraint.Dof.X_TRANSLATION:
            velocity = end_effector.getVelocity()[0]
            linear = True
        elif constraint_dof == EndEffectorConstraint.Dof.Y_TRANSLATION:
            velocity = end_effector.getVelocity()[1]
            linear = True
        elif constraint_dof == EndEffectorConstraint.Dof.Z_TRANSLATION:
            velocity = end_effector.getVelocity()[2]
            linear = True
        elif constraint_dof == EndEffectorConstraint.Dof.X_ROTATION:
            velocity = end_effector.getAngularVelocity()[0]
            linear = False
        elif constraint_dof == EndEffectorConstraint.Dof.Y_ROTATION:
            velocity = end_effector.getAngularVelocity()[1]
            linear = False
        elif constraint_dof == EndEffectorConstraint.Dof.Z_ROTATION:
            velocity = end_effector.getAngularVelocity()[2]
            linear = False
        else:
            logger.error("Unexpected EndEffectorConstraint.Dof.")

        return velocity, linear

    def rescale_velocity(self, velocity, current_velocity, dt, linear):
        """Rescales velocity according to velocity and acceleration limits. Note that this is done DoF-wise only.
        :param float velocity: Action from Gym interface.
        :param float current_velocity: Current velocity of the end-effector.
        :param float dt: Action time-step.
        :param bool linear: Boolean to differentiate between linear and angular scaling.
        :return: Rescaled velocity
        """
        if linear:
            logger.debug(f'Current linear velocity: {current_velocity} m/s')
            logger.debug(f'Initial target velocity: {velocity} m/s')
            max_acceleration = self.max_acceleration
            max_velocity = self.max_velocity
        else:
            logger.debug(f'Current angular velocity: {current_velocity} rad/s')
            logger.debug(f'Initial target velocity: {velocity} rad/s')
            max_acceleration = self.max_angular_acceleration
            max_velocity = self.max_angular_velocity
        if abs((velocity - current_velocity) / dt) > max_acceleration:
            velocity = current_velocity + np.sign(velocity - current_velocity) * (max_acceleration * dt)
        if abs(velocity) > max_velocity:
            velocity = max_velocity * np.sign(velocity)
        logger.debug(f'Rescaled target velocity: {velocity}')

        if math.isnan(velocity):
            logger.error('Unexpected NaN value for velocity, keeping current velocity.')
            velocity = current_velocity
        return velocity

    def rescale_compliance(self, compliance):
        """Rescales compliance between limits defined at initialization of end-effector object.
        :param float compliance: Action from Gym interface.
        :return: Rescaled compliance.
        """
        # Assumes an action range between -1 and 1
        return (compliance + 1) / 2 * (self.max_compliance - self.min_compliance) + self.min_compliance
