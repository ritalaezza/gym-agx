import agx
import agxSDK

import logging

try:
    import matplotlib.pyplot as plt
except:
    print("Could not find matplotlib.")
    plt = None

logger = logging.getLogger('gym_agx.utils')


class CameraConfig:
    def __init__(self, eye, center, up, light_position, light_direction):
        self.camera_pose = {'eye': eye,
                            'center': center,
                            'up': up}
        self.light_pose = {'light_position': light_position,
                           'light_direction': light_direction}


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


class ContactEventListenerRigidBody(agxSDK.ContactEventListener):
    def __init__(self, name, rigid_body, objects_to_ignore):
        """Contact event listener for rigid-body.
        :param String name: agx name for the event listener
        :param agx.RigidBody rigid_body: the rigid-body for which the collision is checked
        :param list objects_to_ignore: list of strings of geometry names to ignore, does not change actual contact only
        if the event listener ignores it
        :return: Bool for collision
        """
        super().__init__(agxSDK.ContactEventListener.ALL)
        self.contactState = False
        self.setName(name)
        self.objects_to_ignore = objects_to_ignore
        self.objects_to_ignore_range = range(len(self.objects_to_ignore))
        self.setFilter(agxSDK.RigidBodyFilter(rigid_body))

    def impact(self, time, contact):
        for i in self.objects_to_ignore_range:
            if contact.geometry(0).getName() == self.objects_to_ignore[i] or contact.geometry(1).getName() == \
                    self.objects_to_ignore[i]:
                return agxSDK.ContactEventListener.KEEP_CONTACT
        self.contactState = True
        return agxSDK.ContactEventListener.KEEP_CONTACT

    def contact(self, time, contact):
        for i in self.objects_to_ignore_range:
            if contact.geometry(0).getName() == self.objects_to_ignore[i] or contact.geometry(1).getName() == \
                    self.objects_to_ignore[i]:
                return agxSDK.ContactEventListener.KEEP_CONTACT
        self.contactState = True
        return agxSDK.ContactEventListener.KEEP_CONTACT

    def separation(self, time, contact_pair):
        for i in self.objects_to_ignore_range:
            if contact_pair.first().getName() == self.objects_to_ignore[i] or contact_pair.second().getName() == \
                    self.objects_to_ignore[i]:
                return
        self.contactState = False
