"""
This module adds a randomly generated goal configuration to the PushRope environment.
"""
import agx
import agxCable

import random
import math


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


def add_goal(sim, logger):
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
    sim.add(rope)

    # Set rope material
    material_rope = rope.getMaterial()
    material_rope.setName("rope_material")
    bulk_material = material_rope.getBulkMaterial()
    bulk_material.setDensity(ROPE_DENSITY)
    surface_material = material_rope.getSurfaceMaterial()
    surface_material.setRoughness(ROPE_ROUGHNESS)
    surface_material.setAdhesion(ROPE_ADHESION, 0)

    # simulate for a short while without graphics to smoothen out kinks at the routing nodes
    for _ in range(500):
        sim.stepForward()

    rope_segments = 0

    rbs = rope.getRigidBodies()
    for i, rb in enumerate(rbs):
        rope_segments += 1
        rbs[i].setName('dlo_' + str(i + 1) + '_goal')
        rb.setMotionControl(agx.RigidBody.STATIC)
        rb_geometries = rb.getGeometries()
        rb_geometries[0].setEnableCollisions(False)

    return rope.getCurrentLength(), rope_segments
