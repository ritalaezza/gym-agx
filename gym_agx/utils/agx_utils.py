import agx
import agxCollide
import agxSDK
import agxIO


# TODO: Is it really good practice to have classes being defined inside methods (instead of just instantiated)?


def create_help_text(sim, app, text_table=None):
    """Write help text indicating how to start simulation textTable is a table with strings that will be drawn above
     the default text
    :param sim: AGX Simulation object
    :param app: OSG Example Application object
    :param text_table:
    """
    class HelpListener(agxSDK.StepEventListener):
        def __init__(self, sim, app, text_table):
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

    sim.add(HelpListener(sim, app, text_table))


def create_body(sim, shape, **args):
    """Helper function that creates a RigidBody according to the given definition.
    Returns the body itself, it's geometry and the OSG node that was created for it.
    :param sim: AGX Simulation object
    :param shape: shape of object - agxCollide.Shape.
    :param args: The definition contains the following parts:
    name - string. Optional. Defaults to "". The name of the new body.
    geometryTransform - agx.AffineMatrix4x4. Optional. Defaults to identity transformation. The local transformation of
    the shape relative to the body.
    motionControl - agx.RigidBody.MotionControl. Optional. Defaults to DYNAMICS.
    material - agx.Material. Optional. Ignored if not given. Material assigned to the geometry created for the body.
    :return: body, geometry
    """
    geometry = agxCollide.Geometry(shape)

    if "geometryTransform" not in args.keys():
        geometry_transform = agx.AffineMatrix4x4()
    else:
        geometry_transform = args["geometryTransform"]

    if "name" in args.keys():
        body = agx.RigidBody(args["name"])
    else:
        body = agx.RigidBody("")

    body.add(geometry, geometry_transform)

    if "position" in args.keys():
        body.setPosition(args["position"])

    if "motionControl" in args.keys():
        body.setMotionControl(args["motionControl"])

    if "material" in args.keys():
        geometry.setMaterial(args["material"])

    sim.add(body)

    return body, geometry


def create_info_printer(sim, app, text_table=None, text_color=None):
    """Write information to screen from lambda functions during the simulation
    :param sim: AGX Simulation object
    :param app: OSG Example Application object
    :param text_table: table containing text fields
    :param text_color: color of text
    """
    class InfoPrinter(agxSDK.StepEventListener):
        def __init__(self, sim, app, text_table, text_color):
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

    sim.add(InfoPrinter(sim, app, text_table, text_color))


def save_sim_file(sim, file):
    """Write information to screen from lambda functions during the simulation
    :param sim: AGX simulation object
    :param file: file directory
    """
    if not agxIO.writeFile(file, sim):
        print("Unable to save simulation!")


def get_state(sim):
    """Return dictionary with object list.
    :param sim: AGX simulation object
    :return: dictionary with rigid objects
    """
    rbs = sim.getRigidBodies()
    state = dict(list(enumerate(rbs)))
    return state


def ctrl_set_action(sim, pos_ctrl, rot_ctrl, grip_ctrl=None):
    """Apply action to simulation.
    :param sim: AGX simulation object
    :param pos_ctrl: Displacement of gripper in x,y,z coordinates
    :param rot_ctrl: Rotation of gripper around x,y,z axes
    :param grip_ctrl: (optional) Displacement of gripper along DLO
    """
    gripper = sim.getObject("gripper")
    gripper_position = gripper.getPosition()
    gripper_rotation = gripper.getRotation()
    gripper.setPosition(gripper_position + pos_ctrl)
    gripper.setRotation(gripper_rotation + rot_ctrl)
