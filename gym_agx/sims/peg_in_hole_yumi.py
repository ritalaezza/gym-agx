"""Simulation of rope pushing

This module creates the simulation files which will be used in PushRope environments.
TODO: Instead of setting all parameters in this file, there should be a parameter file (e.g. YAML or XML).
"""
# AGX Dynamics imports
import agx
import agxPython
import agxCollide
import agxSDK
import agxCable
import agxIO
import agxOSG
import agxRender
import agxUtil

import math
# Python modules
import sys
import logging
import os

# Local modules
from gym_agx.utils.agx_utils import create_body, create_locked_prismatic_base, save_simulation, save_goal_simulation
from gym_agx.utils.agx_classes import KeyboardMotorHandler
from yumi import build_yumi

logger = logging.getLogger('gym_agx.sims')

FILE_NAME = "peg_in_hole_yumi"
# Simulation parameters
TIMESTEP = 1 / 800
N_SUBSTEPS = 20
GRAVITY = True

# Rope parameters
RADIUS = 0.00975  # meters
RESOLUTION = 1/(1.1*RADIUS)  # segments per meter
PEG_POISSON_RATIO = 0.1  # no unit
YOUNG_MODULUS_BEND = 1e5  # 1e4
YOUNG_MODULUS_TWIST = 1e6  # 1e10
YOUNG_MODULUS_STRETCH = 1e8  # Pascals
LENGTH = 0.15
# Aluminum Parameters
ALUMINUM_POISSON_RATIO = 0.35  # no unit
ALUMINUM_YOUNG_MODULUS = 69e9  # Pascals
ALUMINUM_YIELD_POINT = 5e7  # Pascals

# Meshes and textures
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.split(FILE_DIR)[0]
MESH_HOLLOW_CYLINDER_FILE = os.path.join(PACKAGE_DIR, "envs/assets/meshes/mesh_hollow_cylinder.obj")

# Ground Parameters
GROUND_ROUGHNESS = 0.1
GROUND_ADHESION = 1e2
GROUND_ADHESION_OVERLAP = 0
# NOTE: At this overlap, no force is applied. At lower overlap, the adhesion force will work, at higher overlap, the
# (usual) contact forces will be applied
# Rendering Parameters
GROUND_LENGTH_X = LENGTH
GROUND_LENGTH_Y = LENGTH
GROUND_WIDTH = 0.001  # meters
CABLE_GRIPPER_RATIO = 2
SIZE_GRIPPER = CABLE_GRIPPER_RATIO * RADIUS
EYE = agx.Vec3(1, 0.2, 0.3)
CENTER = agx.Vec3(0.25, 0.0, 0.0)
UP = agx.Vec3(0., 0., 1.)

JOINT_NAMES_REV = ['yumi_joint_1_r', 'yumi_joint_2_r', 'yumi_joint_7_r', 'yumi_joint_3_r', 'yumi_joint_4_r',
                   'yumi_joint_5_r', 'yumi_joint_6_r']
# TODO change to other pose
JOINT_INIT_POS = [1.0312985036102147, -0.5908258866009387, -0.2632683336871915, 0.9501280245595223, -1.1971255319558838,
                  1.5739287264532011, 1.598157682864094, 0.005, 0.005, -0.07832728082503805, -2.085892606165076,
                  1.7313167841682604, 0.656188596211846, 2.1706966966564973, 0.6837234666951671, 0.2925179128574373,
                  0.0, 0.0]

def add_rendering(sim):
    app = agxOSG.ExampleApplication(sim)

    # Set renderer
    app.setAutoStepping(True)
    app.setEnableDebugRenderer(False)
    app.setEnableOSGRenderer(True)

    # Create scene graph for rendering
    root = app.getSceneRoot()
    rbs = sim.getRigidBodies()
    for rb in rbs:
        node = agxOSG.createVisual(rb, root)
        if rb.getName() == "hollow_cylinder":
            agxOSG.setDiffuseColor(node, agxRender.Color_SteelBlue())
            agxOSG.setShininess(node, 15)
        elif "dlo" in  rb.getName():  # Cable segments
            agxOSG.setDiffuseColor(node, agxRender.Color(0.0, 1.0, 0.0, 1.0))
        else:
            agxOSG.setDiffuseColor(node, agxRender.Color.Beige())
            agxOSG.setAlpha(node, 0.0)

    # Set rendering options
    scene_decorator = app.getSceneDecorator()
    scene_decorator.setEnableLogo(False)
    scene_decorator.setBackgroundColor(agxRender.Color(1.0, 1.0,1.0, 1.0))

    return app


def build_simulation():
    # Instantiate a simulation
    sim = agxSDK.Simulation()

    # build yumi into scene
    build_yumi(sim, JOINT_INIT_POS)

    # By default the gravity vector is 0,0,-9.81 with a uniform gravity field. (we CAN change that
    # too by creating an agx.PointGravityField for example).
    # AGX uses a right-hand coordinate system (That is Z defines UP. X is right, and Y is into the screen)
    if not GRAVITY:
        logger.info("Gravity off.")
        g = agx.Vec3(0, 0, 0)  # remove gravity
        sim.setUniformGravity(g)

    # Get current delta-t (timestep) that is used in the simulation?
    dt = sim.getTimeStep()
    logger.debug("default dt = {}".format(dt))

    # Change the timestep
    sim.setTimeStep(TIMESTEP)

    # Confirm timestep changed
    dt = sim.getTimeStep()
    logger.debug("new dt = {}".format(dt))

    # Define materials
    material_hard = agx.Material("Aluminum")
    material_hard_bulk = material_hard.getBulkMaterial()
    material_hard_bulk.setPoissonsRatio(ALUMINUM_POISSON_RATIO)
    material_hard_bulk.setYoungsModulus(ALUMINUM_YOUNG_MODULUS)

    # Create hollow cylinde with hole
    scaling_cylinder = agx.Matrix3x3(agx.Vec3(0.0275))
    reader = agxIO.MeshReader()
    hullMesh = agxUtil.createTrimesh(MESH_HOLLOW_CYLINDER_FILE, 0, scaling_cylinder)
    hullGeom = agxCollide.Geometry(hullMesh, agx.AffineMatrix4x4.rotate(agx.Vec3(0, 1, 0), agx.Vec3(0, 0, 1)))

    hollow_cylinder = agx.RigidBody("hollow_cylinder")
    hollow_cylinder.add(hullGeom)
    hollow_cylinder.setMotionControl(agx.RigidBody.STATIC)
    hullGeom.setMaterial(material_hard)
    hollow_cylinder.setPosition(agx.Vec3(0.3, 0, 0.05))
    sim.add(hollow_cylinder)

    # Create rope and set name + properties
    peg = agxCable.Cable(RADIUS, RESOLUTION)
    peg.setName("DLO")
    material_peg = peg.getMaterial()
    peg_material = material_peg.getBulkMaterial()
    peg_material.setPoissonsRatio(PEG_POISSON_RATIO)
    properties = peg.getCableProperties()
    properties.setYoungsModulus(YOUNG_MODULUS_BEND, agxCable.BEND)
    properties.setYoungsModulus(YOUNG_MODULUS_TWIST, agxCable.TWIST)
    properties.setYoungsModulus(YOUNG_MODULUS_STRETCH, agxCable.STRETCH)

    # Add connection between cable and gripper
    tf_0 = agx.AffineMatrix4x4()
    tf_0.setTranslate(0.0, 0, 0.135)
    peg.add(agxCable.BodyFixedNode(sim.getRigidBody("gripper_r_base"), tf_0))
    free_pos = sim.getRigidBody("gripper_r_base").getTransform().transformPoint(agx.Vec3(0.0, 0, LENGTH+0.135))
    peg.add(agxCable.FreeNode(free_pos))

    segment_iterator = peg.begin()
    n_segments = peg.getNumSegments()
    for i in range(n_segments):
        if not segment_iterator.isEnd():
            seg = segment_iterator.getRigidBody()
            seg.setAngularVelocityDamping(1e3)
            segment_iterator.inc()

    # Add rope to simulation
    sim.add(peg)
    # Try to initialize rope
    report = peg.tryInitialize()
    if report.successful():
        print("Successful rope initialization.")
    else:
        print(report.getActualError())

    rbs = peg.getRigidBodies()
    for i in range(len(rbs)):
        rbs[i].setName('dlo_' + str(i+1))
        geo = rbs[i].getGeometries()
        for j in range(len(geo)):
            geo[j].setName('dlo')

    # Set rope material
    material_peg = peg.getMaterial()
    material_peg.setName("rope_material")

    contactMaterial = sim.getMaterialManager().getOrCreateContactMaterial(material_hard, material_peg)
    contactMaterial.setYoungsModulus(1e12)
    fm = agx.IterativeProjectedConeFriction()
    fm.setSolveType(agx.FrictionModel.DIRECT)
    contactMaterial.setFrictionModel(fm)

    return sim


def main(args):
    # Build simulation object
    sim = build_simulation()

    # Save simulation to file
    success = save_simulation(sim, FILE_NAME)
    if success:
        logger.debug("Simulation saved!")
    else:
        logger.debug("Simulation not saved!")

    # Render simulation
    app = add_rendering(sim)
    app.init(agxIO.ArgumentParser([sys.executable] + args))
    app.setCameraHome(EYE, CENTER, UP)  # should only be added after app.init
    app.initSimulation(sim, True)  # This changes timestep and Gravity!
    sim.setTimeStep(TIMESTEP)
    if not GRAVITY:
        logger.info("Gravity off.")
        g = agx.Vec3(0, 0, 0)  # remove gravity
        sim.setUniformGravity(g)

    yumi = sim.getAssembly('yumi')
    yumi.getConstraint1DOF(JOINT_NAMES_REV[0]).getMotor1D().setSpeed(float(0.02))
    yumi.getConstraint1DOF(JOINT_NAMES_REV[1]).getMotor1D().setSpeed(float(0.02))
    yumi.getConstraint1DOF(JOINT_NAMES_REV[2]).getMotor1D().setSpeed(float(0.02))

    yumi.getConstraint1DOF(JOINT_NAMES_REV[3]).getMotor1D().setSpeed(float(-0.05))

    n_seconds = 10
    n_steps = int(n_seconds / (TIMESTEP * N_SUBSTEPS))
    for k in range(n_steps):
        app.executeOneStepWithGraphics()

        t = sim.getTimeStamp()
        t_0 = t
        while t < t_0 + TIMESTEP * N_SUBSTEPS:
            sim.stepForward()
            t = sim.getTimeStamp()

if __name__ == '__main__':
    if agxPython.getContext() is None:
        init = agx.AutoInit()
        main(sys.argv)
