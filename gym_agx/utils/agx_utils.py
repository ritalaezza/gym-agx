# Name: agx_utils.py

# Description: contains some utility functions being used in various simulations.
#
# Never assume any dependency modules already being
# imported somewhere else!
import agx
import agxCollide
import agxOSG
import agxSDK
import agxPython


#
# Write help text indicating how to start simulation
# textTable is a table with strings that will be drawn
# above the default text
def createHelpText(sim, app, textTable=None):
    class HelpListener(agxSDK.StepEventListener):
        def __init__(self, sim, app, textTable):
            super().__init__(agxSDK.StepEventListener.PRE_STEP)
            self.textTable = textTable
            self.app = app
            self.row = 31

        def pre(self, t):
            if t > 3.0:

                self.app.getSceneDecorator().setText(self.row, "", agx.Vec4f(1, 1, 1, 1))

                if self.textTable:
                    startRow = self.row - len(self.textTable)
                    for i, v in enumerate(self.textTable):
                        self.app.getSceneDecorator().setText(startRow + i - 1, "", agx.Vec4f(0.3, 0.6, 0.7, 1))

                self.getSimulation().remove(self)

        def addNotification(self):
            if self.textTable:
                startRow = self.row - len(self.textTable)
                for i, v in enumerate(self.textTable):
                    self.app.getSceneDecorator().setText(startRow + i - 1, v, agx.Vec4f(0.3, 0.6, 0.7, 1))

            self.app.getSceneDecorator().setText(self.row, "Press e to start simulation", agx.Vec4f(0.3, 0.6, 0.7, 1))

    sim.add(HelpListener(sim, app, textTable))


#
# Helper function that creates a RigidBody according to the given definition.
# Returns the body itself, it's geometry and the OSG node that was created for it.
# The definition contains the following parts:
#  shape - agxCollide.Shape.
# name - string. Optional. Deaults to "".
# The name of the new body.
# geometryTransform - agx.AffineMatrix4x4. Optional. Defaults to identity transformation.
#                     The local transformation of the shape relative to the body.
# motionControl - agx.RigidBody.MotionControl. Optional. Defaults to DYNAMICS.
# material - agx.Material. Optional. Ignored if not given.
#            Material assigned to the geometry created for the body.
# color - agx.Vec3. Optional. Ignored if not given.
#
def createBody(sim, root, shape, **args):
    geometry = agxCollide.Geometry(shape)

    if "geometryTransform" not in args.keys():
        geometryTransform = agx.AffineMatrix4x4()
    else:
        geometryTransform = args["geometryTransform"]

    if "name" in args.keys():
        body = agx.RigidBody(args["name"])
    else:
        body = agx.RigidBody("")

    body.add(geometry, geometryTransform)

    if "position" in args.keys():
        body.setPosition(args["position"])

    if "motionControl" in args.keys():
        body.setMotionControl(args["motionControl"])

    if "material" in args.keys():
        geometry.setMaterial(args["material"])

    node = agxOSG.createVisual(geometry, root)

    if "color" in args.keys():
        agxOSG.setDiffuseColor(node, args["color"])

    sim.add(body)

    return body, geometry


#
# Write information to screen from lambda functions during the simulation
#
def createInfoPrinter(sim, app, textTable=None, textColor=None):
    class InfoPrinter(agxSDK.StepEventListener):
        def __init__(self, sim, app, textTable, textColor):
            super().__init__(agxSDK.StepEventListener.POST_STEP)
            self.textTable = textTable
            self.textColor = textColor
            self.app = app
            self.row = 31

        def post(self, t):
            if self.textTable:
                color = agx.Vec4(0.3, 0.6, 0.7, 1)
                if self.textColor:
                    color = self.textColor
                for i, v in enumerate(self.textTable):
                    self.app.getSceneDecorator().setText(i, str(v[0]) + " " + v[1](), color)

    sim.add(InfoPrinter(sim, app, textTable, textColor))
