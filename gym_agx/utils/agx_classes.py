import agx
import agxSDK

import os
import logging
import tempfile
import numpy as np
from enum import Enum

try:
    import matplotlib.pyplot as plt
except:
    print("Could not find matplotlib.")
    plt = None

from gym_agx.utils.agx_utils import get_end_effector_state
from agxPythonModules.utils.numpy_utils import create_numpy_array

logger = logging.getLogger('gym_agx.utils')


class EndEffectorConstraint:
    class Dof(Enum):
        X_TRANSLATIONAL = 0,
        Y_TRANSLATIONAL = 1,
        Z_TRANSLATIONAL = 2,
        X_ROTATIONAL = 3,
        Y_ROTATIONAL = 4,
        Z_ROTATIONAL = 5

    def __init__(self, end_effector_dof, compute_forces_enabled, velocity_control, compliance_control, velocity_index,
                 compliance_index):
        """End effector constraint object, defining important parameters.
        :param end_effector_dof: (GDof) degree of freedom of end effector that this constraint controls
        :param compute_forces_enabled: (Boolean) force and torque can be measured
        :param velocity_control: (Boolean) is velocity controlled
        :param compliance_control: (Boolean) is compliance controlled
        :param velocity_index: (int) index of action vector which controls velocity of this constraint's motor
        :param compliance_index: (int) index of action vector which controls compliance of this constraint's motor"""
        self.end_effector_dof = end_effector_dof
        self.velocity_control = velocity_control
        self.compute_forces_enabled = compute_forces_enabled
        self.compliance_control = compliance_control
        self.velocity_index = velocity_index
        self.compliance_index = compliance_index

    @property
    def is_active(self):
        return True if self.velocity_control or self.compliance_control else False


class EndEffector:
    last_action_index = -1

    def __init__(self, name, controllable, observable, max_velocity=1, max_angular_velocity=1, max_acceleration=1,
                 max_angular_acceleration=1, min_compliance=0, max_compliance=1e6):
        """EndEffector class which keeps track of end effector constraints and action indices.
        :param name
        :param controllable
        :param observable
        :param max_velocity
        :param max_angular_velocity
        :param max_acceleration
        :param max_angular_acceleration
        :param max_compliance
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
        velocity_index = None
        compliance_index = None
        if velocity_control:
            self.last_action_index += 1
            velocity_index = self.last_action_index
        if compliance_control:
            self.last_action_index += 1
            compliance_index = self.last_action_index
        end_effector_constraint = EndEffectorConstraint(end_effector_dof, compute_forces_enabled, velocity_control,
                                                        compliance_control, velocity_index, compliance_index)
        self.constraints.update({name: end_effector_constraint})

    def apply_control(self, sim, action, dt):
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
            logger.debug("Received apply_control command for uncontrollable end effector.")

        return control_actions

    def get_velocity(self, sim, constraint_dof):
        end_effector = sim.getRigidBody(self.name)
        if constraint_dof == EndEffectorConstraint.Dof.X_TRANSLATIONAL:
            velocity = end_effector.getVelocity()[0]
            linear = True
        elif constraint_dof == EndEffectorConstraint.Dof.Y_TRANSLATIONAL:
            velocity = end_effector.getVelocity()[1]
            linear = True
        elif constraint_dof == EndEffectorConstraint.Dof.Z_TRANSLATIONAL:
            velocity = end_effector.getVelocity()[2]
            linear = True
        elif constraint_dof == EndEffectorConstraint.Dof.X_ROTATIONAL:
            velocity = end_effector.getAngularVelocity()[0]
            linear = False
        elif constraint_dof == EndEffectorConstraint.Dof.Y_ROTATIONAL:
            velocity = end_effector.getAngularVelocity()[1]
            linear = False
        elif constraint_dof == EndEffectorConstraint.Dof.Z_ROTATIONAL:
            velocity = end_effector.getAngularVelocity()[2]
            linear = False
        else:
            logger.error("Unexpected EndEffectorConstraint.Dof.")

        return velocity, linear

    def get_state(self, sim):
        if self.observable:
            state = []
            for key, constraint in self.constraints.items():
                if constraint.compute_forces_enabled:
                    constraint_state = get_end_effector_state(sim, key).ravel()
                    logger.debug(f"{key} state: {constraint_state}")
                    state.append(constraint_state)
            return np.asarray(state)
        else:
            logger.error("Received get_state command for unobservable end effector.")

    def rescale_velocity(self, velocity, current_velocity, dt, linear):
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
        if abs(velocity - current_velocity) > max_acceleration:
            velocity = current_velocity + np.sign(velocity - current_velocity) * (max_acceleration * dt)
        if abs(velocity) > max_velocity:
            velocity = max_velocity * np.sign(velocity)
        logger.debug(f'Rescaled target velocity: {velocity}')
        return velocity

    def rescale_compliance(self, compliance):
        # Assumes an action range between -1 and 1
        return (compliance + 1) / 2 * (self.max_compliance - self.min_compliance) + self.min_compliance


class CameraSpecs:
    def __init__(self, eye, center, up, light_position, light_direction):
        self.camera_pose = {'eye': eye,
                            'center': center,
                            'up': up}
        self.light_pose = {'light_position': light_position,
                           'light_direction': light_direction}


class ShowImages(agxSDK.StepEventListener):
    def __init__(self, rti_depth, rti_color, size_depth, size_color):
        super().__init__()

        self.rti_depth = rti_depth
        self.rti_color = rti_color
        self.size_color = size_color
        self.size_depth = size_depth

        if plt is not None:
            self.fig, self.ax = plt.subplots(2, figsize=(10, 10))
            self.obj_color = self.ax[0].imshow(np.ones(self.size_color, dtype=np.uint8))
            self.obj_depth = self.ax[1].imshow(np.ones((self.size_depth[0], size_depth[1]), dtype=np.float32), vmin=0,
                                               vmax=1, cmap='gray')

            plt.ion()
            plt.show()

    def post(self, t):
        # Get pointer to the image
        ptr_color = self.rti_color.getImageData()
        image_color = create_numpy_array(ptr_color, self.size_color, np.uint8)

        ptr_depth = self.rti_depth.getImageData()
        image_depth = create_numpy_array(ptr_depth, self.size_depth, np.float32)

        # check that numpy arrays are created correctly
        if image_color is None or image_depth is None:
            return

        if plt is not None:
            self.obj_color.set_data(np.flip(image_color, 0))
            self.obj_depth.set_data(np.flip(np.squeeze(image_depth), 0))
            plt.draw()
            plt.pause(1e-5)
        else:
            print("Max depth buffer value at time {}: {}".format(t, np.max(image_depth)))
            print("Min depth buffer value at time {}: {}".format(t, np.min(image_depth)))

        # save images to disk at second timestep
        if self.getSimulation().getTimeStep() < t < self.getSimulation().getTimeStep() * 3:
            temp_dir = tempfile.mkdtemp(prefix="agxRenderToImage_")
            filename_color = os.path.join(temp_dir, "color.png")
            filename_depth = os.path.join(temp_dir, "depth.png")
            # Try to save color image to disk. This will work.
            if self.rti_color.saveImage(filename_color):
                print("Saving color image as {} succeeded ".format(filename_color))
            else:
                print("Saving color image as {} failed".format(filename_color))
            # Try to save depth image to disk. This will fail.
            if self.rti_depth.saveImage(filename_depth):
                print("Saving depth image as {} succeeded".format(filename_depth))
            else:
                print("Saving depth image as {} failed".format(filename_depth))


class KeyboardMotorHandler(agxSDK.GuiEventListener):
    """General class to control simulations using keyboard.
    """

    def __init__(self, key_motor_maps):
        """Each instance of this class takes a dictionary
        :param dict key_motor_maps: This dictionary of tuples will assign a motor per key and set the desired speed when
        pressed, taking into account desired direction {agxSDK.GuiEventListener.KEY: (motor, speed)}
        :return Boolean handled: indicates success"""
        super().__init__()
        self.key_motor_maps = key_motor_maps

    def keyboard(self, pressed_key, x, y, alt, down):
        handled = False
        for key, motor_map in self.key_motor_maps.items():
            if pressed_key == key:
                if down:
                    motor_map[0].setSpeed(motor_map[1])
                else:
                    motor_map[0].setSpeed(0)
            handled = True

        return handled


class InfoPrinter(agxSDK.StepEventListener):
    def __init__(self, app, text_table, text_color):
        """Write help text. textTable is a table with strings that will be drawn above the default text.
        :param app: OSG Example Application object
        :param text_table: table with text to be printed on screen
        :return: AGX simulation object
        """
        super().__init__(agxSDK.StepEventListener.POST_STEP)
        self.text_table = text_table
        self.text_color = text_color
        self.app = app
        self.row = 31

    def post(self, t):
        if self.textTable:
            color = agx.Vec4(0.3, 0.6, 0.7, 1)
            if self.text_color:
                color = self.text_color
            for i, v in enumerate(self.text_table):
                self.app.getSceneDecorator().setText(i, str(v[0]) + " " + v[1](), color)


class HelpListener(agxSDK.StepEventListener):
    def __init__(self, app, text_table):
        """Write information to screen from lambda functions during the simulation.
        :param app: OSG Example Application object
        :param text_table: table with text to be printed on screen
        :return: AGX simulation object
        """
        super().__init__(agxSDK.StepEventListener.PRE_STEP)
        self.text_table = text_table
        self.app = app
        self.row = 31

    def pre(self, t):
        if t > 3.0:
            self.app.getSceneDecorator().setText(self.row, "", agx.Vec4f(1, 1, 1, 1))
            if self.text_table:
                start_row = self.row - len(self.text_table)
                for i, v in enumerate(self.text_table):
                    self.app.getSceneDecorator().setText(start_row + i - 1, "", agx.Vec4f(0.3, 0.6, 0.7, 1))

            self.getSimulation().remove(self)

    def addNotification(self):
        if self.text_table:
            start_row = self.row - len(self.text_table)
            for i, v in enumerate(self.text_table):
                self.app.getSceneDecorator().setText(start_row + i - 1, v, agx.Vec4f(0.3, 0.6, 0.7, 1))

        self.app.getSceneDecorator().setText(self.row, "Press e to start simulation", agx.Vec4f(0.3, 0.6, 0.7, 1))
