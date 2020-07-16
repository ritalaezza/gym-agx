import agx
import agxSDK

import os
import logging
import tempfile
import numpy as np

try:
    import matplotlib.pyplot as plt
except:
    print("Could not find matplotlib.")
    plt = None

from agxPythonModules.utils.numpy_utils import create_numpy_array

logger = logging.getLogger('gym_agx.utils')


class CameraConfig:
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
