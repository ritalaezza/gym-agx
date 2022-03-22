import pytest
import logging
import re

logger = logging.getLogger('gym_agx.tests')
ENVIRONMENT_IDS = ['BendWire-v0', 'BendWireObstacle-v0', 'PushRope-v0',
                   'PegInHole-v0', 'RubberBand-v0', 'CableClosing-v0']


@pytest.mark.parametrize("environment_id", ENVIRONMENT_IDS)
def test_environment_names(environment_id):
    env_id_re = re.compile(r'^(?:[\w:-]+\/)?([\w:.-]+)-v(\d+)$')
    match = env_id_re.search(environment_id)
    assert match is not None
