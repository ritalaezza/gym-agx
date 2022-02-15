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

# Python modules
import os
import sys
import math
import random
import logging

# Local modules
import agxUtil
from gym_agx.utils.agx_utils import create_body, create_locked_prismatic_base, save_simulation, save_goal_simulation, \
    add_goal_assembly_from_file
from gym_agx.utils.agx_classes import KeyboardMotorHandler

logger = logging.getLogger('gym_agx.sims')

FILE_NAME = "push_rope"

# Simulation parameters
TIMESTEP = 1 / 1000
N_SUBSTEPS = 20
GRAVITY = True

# Rope parameters
RADIUS = 0.0015  # meters
LENGTH = 0.05  # meters
RESOLUTION = 500  # segments per meter
ROPE_POISSON_RATIO = 0.01  # no unit
YOUNG_MODULUS_BEND = 1e3  # 1e5
YOUNG_MODULUS_TWIST = 1e3  # 1e10
YOUNG_MODULUS_STRETCH = 8e9  # Pascals
ROPE_ROUGHNESS = 10
ROPE_ADHESION = 0.001
ROPE_DENSITY = 1.5  # kg/m3
NODE_AMOUNT = 9

# Aluminum Parameters
ALUMINUM_POISSON_RATIO = 0.35  # no unit
ALUMINUM_YOUNG_MODULUS = 69e9  # Pascals
ALUMINUM_YIELD_POINT = 5e7  # Pascals

# Pusher Parameters
PUSHER_RADIUS = 0.005
PUSHER_HEIGHT = 0.02
PUSHER_ROUGHNESS = 1
PUSHER_ADHESION = 0
PUSHER_ADHESION_OVERLAP = 0

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
EYE = agx.Vec3(0, 0, 0.5)
CENTER = agx.Vec3(0, 0, 0)
UP = agx.Vec3(0., 0., 0.)

# Control parameters
MAX_MOTOR_FORCE = 1  # Newtons


def add_rendering(sim):
    camera_distance = 0.5
    light_pos = agx.Vec4(LENGTH / 2, - camera_distance, camera_distance, 1.)
    light_dir = agx.Vec3(0., 0., -1.)

    app = agxOSG.ExampleApplication(sim)

    app.setAutoStepping(True)
    app.setEnableDebugRenderer(False)
    app.setEnableOSGRenderer(True)

    scene_decorator = app.getSceneDecorator()
    light_source_0 = scene_decorator.getLightSource(agxOSG.SceneDecorator.LIGHT0)
    light_source_0.setPosition(light_pos)
    light_source_0.setDirection(light_dir)

    root = app.getRoot()
    rbs = sim.getRigidBodies()
    for rb in rbs:
        name = rb.getName()
        node = agxOSG.createVisual(rb, root)
        if name == "ground":
            agxOSG.setDiffuseColor(node, agxRender.Color.Gray())
        elif name == "pusher":
            agxOSG.setDiffuseColor(node, agxRender.Color(0.0, 0.0, 1.0, 1.0))
        elif "obstacle" in name:
            agxOSG.setDiffuseColor(node, agxRender.Color(1.0, 0.0, 0.0, 1.0))
        elif "dlo" in name:
            agxOSG.setDiffuseColor(node, agxRender.Color(0.0, 1.0, 0.0, 1.0))
        elif "bounding_box" in name:
            agxOSG.setDiffuseColor(node, agxRender.Color.Burlywood())
        else:  # Base segments
            agxOSG.setDiffuseColor(node, agxRender.Color.Beige())
            agxOSG.setAlpha(node, 0.2)
        if "goal" in name:
            agxOSG.setAlpha(node, 0.2)

    scene_decorator = app.getSceneDecorator()
    light_source_0 = scene_decorator.getLightSource(agxOSG.SceneDecorator.LIGHT0)
    light_source_0.setPosition(light_pos)
    light_source_0.setDirection(light_dir)
    scene_decorator.setEnableLogo(False)

    return app


def sample_random_goal(sim, render=False):
    """Goal Randomization: for the PushRope environment it is too difficult to generate proper trajectories that lead to
     varied shapes. For this reason, a new rope is added to the scene every time, and routed through random points
    :param sim: AGX Dynamics simulation object
    :param bool render: toggle rendering for debugging purposes only
    """
    # get goal assembly
    goal_scene = sim.getAssembly("goal_assembly")

    # Create rope
    rope = agxCable.Cable(RADIUS, RESOLUTION)
    rope_z = RADIUS

    # Create random positions for first node
    new_node_x = random.uniform((-0.9 * LENGTH + RADIUS), (0.9 * LENGTH - RADIUS))
    new_node_y = random.uniform((-0.9 * LENGTH + RADIUS), (0.9 * LENGTH - RADIUS))

    # compute angle pointing towards center
    # rope_angle = math.atan2(-new_node_y, -new_node_x)

    # Uniformly distributed initial angle
    rope_angle = random.uniform(-math.pi, math.pi)

    rope.add(agxCable.FreeNode(new_node_x, new_node_y, rope_z))

    # compute length of routing sections
    section_length = LENGTH / (NODE_AMOUNT-1)

    for i in range(NODE_AMOUNT-1):
        # modify previous angle and calculate new node coordinates
        rope_angle += random.gauss(0, math.pi / 4)

        prev_node_x = new_node_x
        prev_node_y = new_node_y

        new_node_x = prev_node_x + math.cos(rope_angle) * section_length
        new_node_y = prev_node_y + math.sin(rope_angle) * section_length

        # if node ends up too close to the borders, reset angle to point towards center
        while abs(new_node_x) / LENGTH > 0.9 or abs(new_node_y) / LENGTH > 0.9:
            # intentionally using the new coordinates for additional randomization
            rope_angle = math.atan2(-new_node_y, -new_node_x)
            rope_angle += random.gauss(0, math.pi / 4)

            new_node_x = prev_node_x + math.cos(rope_angle) * section_length
            new_node_y = prev_node_y + math.sin(rope_angle) * section_length

        rope.add(agxCable.FreeNode(new_node_x, new_node_y, rope_z))

    # Set rope name and properties
    rope.setName("DLO_goal")
    properties = rope.getCableProperties()
    properties.setYoungsModulus(YOUNG_MODULUS_BEND, agxCable.BEND)
    properties.setYoungsModulus(YOUNG_MODULUS_TWIST, agxCable.TWIST)
    properties.setYoungsModulus(YOUNG_MODULUS_STRETCH, agxCable.STRETCH)

    # Try to initialize rope
    report = rope.tryInitialize()
    if report.successful():
        logger.info("Successful rope initialization.")
    else:
        print(report.getActualError())

    # Add rope to simulation
    goal_scene.add(rope)

    start_scene = sim.getAssembly("start_assembly")
    agxUtil.setEnableCollisions(goal_scene, start_scene, False)  # need to disable collisions again after adding rope

    # Set rope material
    material_rope = rope.getMaterial()
    material_rope.setName("rope_material")
    bulk_material = material_rope.getBulkMaterial()
    bulk_material.setDensity(ROPE_DENSITY)
    surface_material = material_rope.getSurfaceMaterial()
    surface_material.setRoughness(ROPE_ROUGHNESS)
    surface_material.setAdhesion(ROPE_ADHESION, 0)

    # simulate for a short while without graphics to smoothen out kinks at the routing nodes
    for _ in range(1000):
        sim.stepForward()

    # reset timestamp, after simulation
    sim.setTimeStamp(0)

    rbs = rope.getRigidBodies()
    for i, rb in enumerate(rbs):
        rbs[i].setName('dlo_' + str(i + 1) + '_goal')
        rb.setMotionControl(agx.RigidBody.STATIC)

    if render:
        # Add keyboard listener
        motor_x = sim.getConstraint1DOF("pusher_joint_base_x").getMotor1D()
        motor_y = sim.getConstraint1DOF("pusher_joint_base_y").getMotor1D()
        motor_z = sim.getConstraint1DOF("pusher_joint_base_z").getMotor1D()
        key_motor_map = {agxSDK.GuiEventListener.KEY_Up: (motor_y, 0.05),
                         agxSDK.GuiEventListener.KEY_Down: (motor_y, -0.05),
                         agxSDK.GuiEventListener.KEY_Right: (motor_x, 0.05),
                         agxSDK.GuiEventListener.KEY_Left: (motor_x, -0.05),
                         65365: (motor_z, 0.05),
                         65366: (motor_z, -0.05)}
        sim.add(KeyboardMotorHandler(key_motor_map))

        # Render simulation
        app = add_rendering(sim)
        app.init(agxIO.ArgumentParser([sys.executable]))  # no args being passed to agxViewer!
        app.setCameraHome(EYE, CENTER, UP)  # should only be added after app.init
        app.initSimulation(sim, True)  # This changes timestep and Gravity!
        sim.setTimeStep(TIMESTEP)
        if not GRAVITY:
            logger.info("Gravity off.")
            g = agx.Vec3(0, 0, 0)  # remove gravity
            sim.setUniformGravity(g)

        n_seconds = 10
        t = sim.getTimeStamp()
        while t < n_seconds:
            app.executeOneStepWithGraphics()

            t = sim.getTimeStamp()
            t_0 = t
            while t < t_0 + TIMESTEP * N_SUBSTEPS:
                sim.stepForward()
                t = sim.getTimeStamp()


def sample_fixed_goal(sim, app=None):
    """Define the trajectory to generate fixed goal. For the PushRope environment a keyboard listener is added to allow
    manual control
    :param sim: AGX Dynamics simulation object
    :param app: AGX Dynamics application object
    """
    # Add keyboard listener
    motor_x = sim.getConstraint1DOF("pusher_goal_joint_base_x").getMotor1D()
    motor_y = sim.getConstraint1DOF("pusher_goal_joint_base_y").getMotor1D()
    motor_z = sim.getConstraint1DOF("pusher_goal_joint_base_z").getMotor1D()
    key_motor_map = {agxSDK.GuiEventListener.KEY_Up: (motor_y, 0.05),
                     agxSDK.GuiEventListener.KEY_Down: (motor_y, -0.05),
                     agxSDK.GuiEventListener.KEY_Right: (motor_x, 0.05),
                     agxSDK.GuiEventListener.KEY_Left: (motor_x, -0.05),
                     65365: (motor_z, 0.05),
                     65366: (motor_z, -0.05)}
    sim.add(KeyboardMotorHandler(key_motor_map))

    n_seconds = 30
    t = sim.getTimeStamp()
    while t < n_seconds:
        if app:
            app.executeOneStepWithGraphics()

        t = sim.getTimeStamp()
        t_0 = t
        while t < t_0 + TIMESTEP * N_SUBSTEPS:
            sim.stepForward()
            t = sim.getTimeStamp()


def build_simulation(goal=False, rope=True):
    """Builds simulations for both start and goal configurations
    :param bool goal: toggles between simulation definition of start and goal configurations
    :param bool rope: add rope to the scene or not
    :return agxSDK.Simulation: simulation object
    """
    assembly_name = "start_"
    goal_string = ""
    if goal:
        assembly_name = "goal_"
        goal_string = "_goal"

    # Instantiate a simulation
    sim = agxSDK.Simulation()

    # By default, the gravity vector is 0,0,-9.81 with a uniform gravity field. (we CAN change that
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

    # Create a new empty Assembly
    scene = agxSDK.Assembly()
    scene.setName(assembly_name + "assembly")

    # Add start assembly to simulation
    sim.add(scene)

    # Define materials
    material_ground = agx.Material("Aluminum")
    bulk_material = material_ground.getBulkMaterial()
    bulk_material.setPoissonsRatio(ALUMINUM_POISSON_RATIO)
    bulk_material.setYoungsModulus(ALUMINUM_YOUNG_MODULUS)
    surface_material = material_ground.getSurfaceMaterial()
    surface_material.setRoughness(GROUND_ROUGHNESS)
    surface_material.setAdhesion(GROUND_ADHESION, GROUND_ADHESION_OVERLAP)

    material_pusher = agx.Material("Aluminum")
    bulk_material = material_pusher.getBulkMaterial()
    bulk_material.setPoissonsRatio(ALUMINUM_POISSON_RATIO)
    bulk_material.setYoungsModulus(ALUMINUM_YOUNG_MODULUS)
    surface_material = material_pusher.getSurfaceMaterial()
    surface_material.setRoughness(PUSHER_ROUGHNESS)
    surface_material.setAdhesion(PUSHER_ADHESION, PUSHER_ADHESION_OVERLAP)

    # Create a ground plane and bounding box to prevent falls
    ground = create_body(name="ground" + goal_string, shape=agxCollide.Box(GROUND_LENGTH_X, GROUND_LENGTH_Y,
                                                                           GROUND_WIDTH),
                         position=agx.Vec3(0, 0, -GROUND_WIDTH / 2),
                         motion_control=agx.RigidBody.STATIC,
                         material=material_ground)
    scene.add(ground)

    bounding_box = create_body(name="bounding_box_1" + goal_string, shape=agxCollide.Box(GROUND_LENGTH_X, GROUND_WIDTH,
                                                                                         RADIUS * 4),
                               position=agx.Vec3(0, GROUND_LENGTH_Y - GROUND_WIDTH, RADIUS * 4 - GROUND_WIDTH),
                               motion_control=agx.RigidBody.STATIC,
                               material=material_ground)
    scene.add(bounding_box)
    bounding_box = create_body(name="bounding_box_2" + goal_string, shape=agxCollide.Box(GROUND_LENGTH_X, GROUND_WIDTH,
                                                                                         RADIUS * 4),
                               position=agx.Vec3(0, - GROUND_LENGTH_Y + GROUND_WIDTH, RADIUS * 4 - GROUND_WIDTH),
                               motion_control=agx.RigidBody.STATIC,
                               material=material_ground)
    scene.add(bounding_box)
    bounding_box = create_body(name="bounding_box_3" + goal_string, shape=agxCollide.Box(GROUND_WIDTH, GROUND_LENGTH_Y,
                                                                                         RADIUS * 4),
                               position=agx.Vec3(GROUND_LENGTH_X - GROUND_WIDTH, 0, RADIUS * 4 - GROUND_WIDTH),
                               motion_control=agx.RigidBody.STATIC,
                               material=material_ground)
    scene.add(bounding_box)
    bounding_box = create_body(name="bounding_box_4" + goal_string, shape=agxCollide.Box(GROUND_WIDTH, GROUND_LENGTH_Y,
                                                                                         RADIUS * 4),
                               position=agx.Vec3(- GROUND_LENGTH_X + GROUND_WIDTH, 0, RADIUS * 4 - GROUND_WIDTH),
                               motion_control=agx.RigidBody.STATIC,
                               material=material_ground)
    scene.add(bounding_box)

    if rope:
        # Create rope
        rope = agxCable.Cable(RADIUS, RESOLUTION)
        rope.add(agxCable.FreeNode(GROUND_LENGTH_X / 2 - RADIUS * 2, GROUND_LENGTH_Y / 2 - RADIUS * 2, RADIUS * 2))
        rope.add(agxCable.FreeNode(GROUND_LENGTH_X / 2 - RADIUS * 2, GROUND_LENGTH_Y / 2 - LENGTH - RADIUS * 2,
                                   RADIUS * 2))

        # Set rope name and properties
        rope.setName("DLO" + goal_string)
        properties = rope.getCableProperties()
        properties.setYoungsModulus(YOUNG_MODULUS_BEND, agxCable.BEND)
        properties.setYoungsModulus(YOUNG_MODULUS_TWIST, agxCable.TWIST)
        properties.setYoungsModulus(YOUNG_MODULUS_STRETCH, agxCable.STRETCH)

        # Try to initialize rope
        report = rope.tryInitialize()
        if report.successful():
            print("Successful rope initialization.")
        else:
            print(report.getActualError())

        # Add rope to simulation
        scene.add(rope)

        rbs = rope.getRigidBodies()
        for i in range(len(rbs)):
            rbs[i].setName('dlo_' + str(i+1) + goal_string)

        # Set rope material
        material_rope = rope.getMaterial()
        material_rope.setName("rope_material")
        bulk_material = material_rope.getBulkMaterial()
        bulk_material.setDensity(ROPE_DENSITY)
        surface_material = material_rope.getSurfaceMaterial()
        surface_material.setRoughness(ROPE_ROUGHNESS)
        surface_material.setAdhesion(ROPE_ADHESION, 0)

        # Check mass
        rope_mass = rope.getMass()
        print("Rope mass: {}".format(rope_mass))

        # Create contact materials
        contact_material_ground_rope = sim.getMaterialManager().getOrCreateContactMaterial(material_ground,
                                                                                           material_rope)
        contact_material_pusher_rope = sim.getMaterialManager().getOrCreateContactMaterial(material_pusher,
                                                                                           material_rope)
        contact_material_ground_rope.setUseContactAreaApproach(True)
        sim.add(contact_material_ground_rope)
        sim.add(contact_material_pusher_rope)

    rotation_cylinder = agx.OrthoMatrix3x3()
    rotation_cylinder.setRotate(agx.Vec3.Y_AXIS(), agx.Vec3.Z_AXIS())

    pusher = create_body(name="pusher" + goal_string,
                         shape=agxCollide.Cylinder(PUSHER_RADIUS, PUSHER_HEIGHT),
                         position=agx.Vec3(0.0, 0.0, PUSHER_HEIGHT / 2),
                         rotation=rotation_cylinder,
                         motion_control=agx.RigidBody.DYNAMICS,
                         material=material_pusher)
    scene.add(pusher)

    # Create base for pusher motors
    prismatic_base = create_locked_prismatic_base("pusher" + goal_string, pusher.getRigidBody("pusher" + goal_string),
                                                  position_ranges=[(-GROUND_LENGTH_X, GROUND_LENGTH_X),
                                                                   (-GROUND_LENGTH_Y, GROUND_LENGTH_Y),
                                                                   (0., 3 * RADIUS)],
                                                  motor_ranges=[(-MAX_MOTOR_FORCE, MAX_MOTOR_FORCE),
                                                                (-MAX_MOTOR_FORCE, MAX_MOTOR_FORCE),
                                                                (-MAX_MOTOR_FORCE, MAX_MOTOR_FORCE)])

    scene.add(prismatic_base)

    return sim


# Build and save scene to file
def main(args):
    # 1) Build start simulation object
    sim = build_simulation()

    # Save start simulation to file
    success = save_simulation(sim, FILE_NAME)
    if not success:
        logger.debug("Simulation not saved!")

    # 2) Build goal simulation object
    goal_sim = build_simulation(goal=True)

    # Render simulation
    app = add_rendering(goal_sim)
    app.init(agxIO.ArgumentParser([sys.executable] + args))
    app.setCameraHome(EYE, CENTER, UP)  # should only be added after app.init
    app.initSimulation(goal_sim, True)  # This changes timestep and Gravity!
    goal_sim.setTimeStep(TIMESTEP)
    if not GRAVITY:
        logger.info("Gravity off.")
        g = agx.Vec3(0, 0, 0)  # remove gravity
        goal_sim.setUniformGravity(g)

    # 3) Sample fixed goal
    sample_fixed_goal(goal_sim, app)

    # Set goal objects to static
    rbs = goal_sim.getRigidBodies()
    for rb in rbs:
        name = rb.getName()
        if "_goal" in name:
            rb.setMotionControl(agx.RigidBody.STATIC)

    # Save fixed goal simulation to file
    success = save_simulation(goal_sim, FILE_NAME + "_goal")
    if not success:
        logger.debug("Fixed goal simulation not saved!")

    # 4) Build random goal simulation object (without DLO)
    random_goal_sim = build_simulation(goal=True, rope=False)

    success = save_simulation(random_goal_sim, FILE_NAME + "_goal_random", aagx=True)
    if not success:
        logger.debug("Goal simulation not saved!")

    file_directory = os.path.dirname(os.path.abspath(__file__))
    package_directory = os.path.split(file_directory)[0]
    random_goal_file = os.path.join(package_directory, 'envs/assets',  FILE_NAME + "_goal_random.agx")
    add_goal_assembly_from_file(sim, random_goal_file)

    # Test random goal generation
    sample_random_goal(sim, render=True)


if __name__ == '__main__':
    if agxPython.getContext() is None:
        init = agx.AutoInit()
        main(sys.argv)
