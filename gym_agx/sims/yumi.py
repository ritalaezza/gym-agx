# Creates simulation file for the yumi robot

import agx
import agxPython
import agxCollide
import agxRender
import agxSDK
import agxIO
import agxOSG
import agxModel

# Python modules
import logging
import math
import sys
import os
from gym_agx.utils.agx_utils import create_body, save_simulation
from gym_agx.utils.agx_utils_optimize import createConvexDecomposition, reduceMesh

logger = logging.getLogger('gym_agx.sims')

FILE_NAME = 'yumi_robot_base_file'

FILE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIRECTORY = os.path.split(FILE_DIRECTORY)[0]
DESCRIPTION_PATH = os.path.join(PACKAGE_DIRECTORY, 'envs', 'assets')
URDF_PATH = os.path.join(DESCRIPTION_PATH, 'yumi_description', 'urdf', 'yumi.urdf')
GRAVITY = True
LENGTH = 1  # defined as half the total length
HEIGHT = 0.05  # defined as half the total height
N_SUBSTEPS = 2
TIMESTEP = 1 / 100  # seconds (eq. 100 Hz)
EYE = agx.Vec3(LENGTH / 2, -5 * LENGTH, 0)
CENTER = agx.Vec3(LENGTH / 2, 0, 0)
UP = agx.Vec3(0., 0., 1.)

INIT_JOINT_POS = [0.7, -1.7, -0.8, 1.0, -2.2, 1.0, 0.0, 0.0, 0.0, -0.7, -1.7, 0.8, 1.0, 2.2, 1.0, 0.0, 0.0, 0.0]

# urdf name for joints
JOINT_NAMES_REV = ['yumi_joint_1_l', 'yumi_joint_2_l', 'yumi_joint_7_l', 'yumi_joint_3_l', 'yumi_joint_4_l',
                      'yumi_joint_5_l', 'yumi_joint_6_l',
                      'yumi_joint_1_r', 'yumi_joint_2_r', 'yumi_joint_7_r', 'yumi_joint_3_r', 'yumi_joint_4_r',
                      'yumi_joint_5_r', 'yumi_joint_6_r']

JOINT_EFFORT_REV = [45, 35, 30, 25, 25, 25, 25, 45, 35, 30, 25, 25, 25, 25]  # maximum joint effort, assuming same force in
                                                                    # upper and lower, same order as jointNamesRevolute

GRIPPER_EFFORT = 15  # set the grip force
JOINT_NAME_GRIPPER = ['gripper_l_joint', 'gripper_l_joint_m', 'gripper_r_joint',
                      'gripper_r_joint_m']  # name of gripper joints in urdf

# How to set up contact material for grasping.
'''
# contact material for grippers
MaterialDescription = {
    'friction': 0.3,  # Friction value (between 0..1).
    'frictionModel': agx.FrictionModel.DIRECT,
    # Use the direct solver for stable grasping - computationally more expensive, but necessary here.
    'restitution': 0.0,  # Coefficient of restitution. Leave it at zero here.
    'useContactArea': True,
    # Use the more detailed contact area approach for calculating the contact forces. Default is false.
    'maxElasticRestLength': 1E-3,
    # Let maximum 0.02m of the contact rest length be in the elastic domain (solid above that).
    'youngsModulus': 1e10
    # Stiffness of the contact, 200GPa would be Steel-Steel, but this grasping material is softer.
}


def setupContactMaterial(sim):
    material1 = agx.Material("gripperMaterial")
    material2 = agx.Material("objectMaterial")

    cm = sim.getMaterialManager().getOrCreateContactMaterial(material1, material2)
    fm = agx.IterativeProjectedConeFriction()
    fm.setSolveType(MaterialDescription['frictionModel'])

    # Now setup the material properties from the description
    cm.setFrictionCoefficient(MaterialDescription['friction'])
    cm.setRestitution(MaterialDescription['restitution'])
    cm.setUseContactAreaApproach(MaterialDescription['useContactArea'])
    cm.setMinMaxElasticRestLength(cm.getMinElasticRestLength(), MaterialDescription['maxElasticRestLength'])
    cm.setFrictionModel(fm)
    cm.setYoungsModulus(MaterialDescription['youngsModulus'])

    return material1, material2
'''


def optimize(rigid_bodies):
    render_data_size = 0
    mesh_data_size = 0
    for b in rigid_bodies:
        for g in b.getGeometries():
            if not g.getEnableCollisions():
                print("Not enabled for collision, skipped: ", g.getName())
                continue
            if "gripper" in b.getName():
                size, size_mesh = createConvexDecomposition(g)
            else:
                size, size_mesh = reduceMesh(g)
            render_data_size += size
            mesh_data_size += size_mesh
    print("Total render data size {} Mb".format(render_data_size/1E6))
    print("Total mesh data size {} Mb".format(mesh_data_size/1E6))


def collision_between_bodies(rb1, rb2, collision=True):
    for i in range(len(rb1.getGeometries())):
        for j in range(len(rb2.getGeometries())):
            rb1.getGeometries()[i].setEnableCollisions(rb2.getGeometries()[j], collision)


def add_rendering(sim):
    camera_distance = 0.5
    light_pos = agx.Vec4(LENGTH / 2, - camera_distance, camera_distance, 1.)
    light_dir = agx.Vec3(0., 0., -1.)

    app = agxOSG.ExampleApplication(sim)

    app.setAutoStepping(False)
    app.setEnableDebugRenderer(False)
    app.setEnableOSGRenderer(True)

    root = app.getSceneRoot()
    rbs = sim.getRigidBodies()
    for rb in rbs:
        if rb.getName() == "ground":
            ground_node = agxOSG.createVisual(rb, root)
            agxOSG.setDiffuseColor(ground_node, agxRender.Color.Gray())
        else:  # yumi
            yumi_node = agxOSG.createVisual(rb, root)
            #agxOSG.setDiffuseColor(yumi_node, agxRender.Color.Blue())

    scene_decorator = app.getSceneDecorator()
    light_source_0 = scene_decorator.getLightSource(agxOSG.SceneDecorator.LIGHT0)
    light_source_0.setPosition(light_pos)
    light_source_0.setDirection(light_dir)
    scene_decorator.setEnableLogo(False)

    return app


def build_simulation():
    # Instantiate a simulation
    sim = agxSDK.Simulation()

    build_yumi(sim, INIT_JOINT_POS)

    return sim


def build_yumi(sim, init_joint_pos_):
    # ---------  Create a ground plane -----------------------------------
    ground = create_body(name="ground", shape=agxCollide.Box(LENGTH, LENGTH, HEIGHT),
                         position=agx.Vec3(0, 0, -HEIGHT),
                         motion_control=agx.RigidBody.STATIC)
    sim.add(ground)

    # ------------- YuMi --------------------------------------------------

    # initial joint position
    init_joint_pos = agx.RealVector()
    for i in range(len(init_joint_pos_)):
        init_joint_pos.append(init_joint_pos_[i])

    # read urdf
    yumi_assembly_ref = agxModel.UrdfReader.read(URDF_PATH, DESCRIPTION_PATH, init_joint_pos, True)
    if yumi_assembly_ref.get() == None:
        print("Error reading the URDF file.")
        sys.exit(2)

    yumi = yumi_assembly_ref.get()

    # optimize geometry
    optimize(yumi_assembly_ref.getRigidBodies())

    # Add the yumi assembly to the simulation and create visualization for it
    sim.add(yumi_assembly_ref.get())

    # Enable Motor1D (speed controller) on all revolute joints and set effort limits
    for i in range(len(JOINT_NAMES_REV)):
        yumi.getConstraint1DOF(JOINT_NAMES_REV[i]).getMotor1D().setEnable(True)
        yumi.getConstraint1DOF(JOINT_NAMES_REV[i]).getMotor1D().setForceRange(-JOINT_EFFORT_REV[i], JOINT_EFFORT_REV[i])
        yumi.getConstraint1DOF(JOINT_NAMES_REV[i]).getMotor1D().setSpeed(float(0.0))

    # Enable Motor1D (speed controller) on all prismatic joints (grippers) and set effort limits
    for i in range(len(JOINT_NAME_GRIPPER)):
        yumi.getConstraint1DOF(JOINT_NAME_GRIPPER[i]).getMotor1D().setEnable(True)
        yumi.getConstraint1DOF(JOINT_NAME_GRIPPER[i]).getMotor1D().setForceRange(-GRIPPER_EFFORT, GRIPPER_EFFORT)

    # collision between floor and root
    collision_between_bodies(sim.getAssembly('ground').getRigidBody('ground'),
                             sim.getAssembly('yumi').getRigidBody('yumi_body'), False)

    # disable collision between connected links.
    collision_between_bodies(yumi_assembly_ref.getRigidBody('yumi_body'),
                             yumi_assembly_ref.getRigidBody('yumi_link_1_r'), False)
    collision_between_bodies(yumi_assembly_ref.getRigidBody('yumi_body'),
                             yumi_assembly_ref.getRigidBody('yumi_link_1_l'), False)

    collision_between_bodies(yumi_assembly_ref.getRigidBody('yumi_link_1_r'),
                             yumi_assembly_ref.getRigidBody('yumi_link_2_r'), False)
    collision_between_bodies(yumi_assembly_ref.getRigidBody('yumi_link_1_l'),
                             yumi_assembly_ref.getRigidBody('yumi_link_2_l'), False)

    collision_between_bodies(yumi_assembly_ref.getRigidBody('yumi_link_2_r'),
                             yumi_assembly_ref.getRigidBody('yumi_link_3_r'), False)
    collision_between_bodies(yumi_assembly_ref.getRigidBody('yumi_link_2_l'),
                             yumi_assembly_ref.getRigidBody('yumi_link_3_l'), False)

    collision_between_bodies(yumi_assembly_ref.getRigidBody('yumi_link_3_r'),
                             yumi_assembly_ref.getRigidBody('yumi_link_4_r'), False)
    collision_between_bodies(yumi_assembly_ref.getRigidBody('yumi_link_3_l'),
                             yumi_assembly_ref.getRigidBody('yumi_link_4_l'), False)

    collision_between_bodies(yumi_assembly_ref.getRigidBody('yumi_link_4_r'),
                             yumi_assembly_ref.getRigidBody('yumi_link_5_r'), False)
    collision_between_bodies(yumi_assembly_ref.getRigidBody('yumi_link_4_l'),
                             yumi_assembly_ref.getRigidBody('yumi_link_5_l'), False)

    collision_between_bodies(yumi_assembly_ref.getRigidBody('yumi_link_5_r'),
                             yumi_assembly_ref.getRigidBody('yumi_link_6_r'), False)
    collision_between_bodies(yumi_assembly_ref.getRigidBody('yumi_link_5_l'),
                             yumi_assembly_ref.getRigidBody('yumi_link_6_l'), False)

    collision_between_bodies(yumi_assembly_ref.getRigidBody('yumi_link_6_r'),
                             yumi_assembly_ref.getRigidBody('yumi_link_7_r'), False)
    collision_between_bodies(yumi_assembly_ref.getRigidBody('yumi_link_6_l'),
                             yumi_assembly_ref.getRigidBody('yumi_link_7_l'), False)

    collision_between_bodies(yumi_assembly_ref.getRigidBody('yumi_link_6_r'),
                             yumi_assembly_ref.getRigidBody('yumi_link_7_r'), False)
    collision_between_bodies(yumi_assembly_ref.getRigidBody('yumi_link_6_l'),
                             yumi_assembly_ref.getRigidBody('yumi_link_7_l'), False)

    collision_between_bodies(yumi_assembly_ref.getRigidBody('yumi_link_7_r'),
                             yumi_assembly_ref.getRigidBody('gripper_r_base'), False)
    collision_between_bodies(yumi_assembly_ref.getRigidBody('yumi_link_7_l'),
                             yumi_assembly_ref.getRigidBody('gripper_l_base'), False)

    collision_between_bodies(yumi_assembly_ref.getRigidBody('gripper_r_base'),
                             yumi_assembly_ref.getRigidBody('gripper_r_finger_r'), False)
    collision_between_bodies(yumi_assembly_ref.getRigidBody('gripper_r_base'),
                             yumi_assembly_ref.getRigidBody('gripper_r_finger_l'), False)
    collision_between_bodies(yumi_assembly_ref.getRigidBody('gripper_r_finger_r'),
                             yumi_assembly_ref.getRigidBody('gripper_r_finger_l'), False)

    collision_between_bodies(yumi_assembly_ref.getRigidBody('gripper_l_base'),
                             yumi_assembly_ref.getRigidBody('gripper_l_finger_r'), False)
    collision_between_bodies(yumi_assembly_ref.getRigidBody('gripper_l_base'),
                             yumi_assembly_ref.getRigidBody('gripper_l_finger_l'), False)
    collision_between_bodies(yumi_assembly_ref.getRigidBody('gripper_l_finger_r'),
                             yumi_assembly_ref.getRigidBody('gripper_l_finger_l'), False)

    sim.getDynamicsSystem().setEnableContactWarmstarting(True)



# Build and save scene to file
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


    n_seconds = 40
    n_steps = int(n_seconds / (TIMESTEP * N_SUBSTEPS))

    count = 0
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
