def idle():
    return 0

def left():
    return - agx.Vec3(step, 0, 0)

def right():
    return agx.Vec3(step, 0, 0)

def up():
    return agx.Vec3(0, 0, step)

def down():
    return - agx.Vec3(0, 0, step)


def executeAction(type,argument,step):
    if (type == "position"):
        actions = {
            0: idle,
            1: left,
            2: right,
            3: up,
            4: down,
        }
    else if (type == "velocity"):
        actions = {
        }
    return actions.get(argument, "Invalid action")(step)
