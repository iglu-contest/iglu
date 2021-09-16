from collections import Counter
import logging
import numpy as np
import gym
from typing import Union

from minerl_patched.herobraine.hero.handlers.agent.action import Action
from minerl_patched.herobraine.hero import spaces
import minerl_patched.herobraine.hero.handlers as handlers
import minerl_patched.herobraine.hero.handlers.agent.observations.location_stats as loc_obs

from .const import BUILD_ZONE_SIZE, \
                   GROUND_LEVEL, \
                   block_map, block2id, block_short2id
from .tasks import TaskSet, RandomTasks

logger = logging.getLogger(__name__)


class HotBarChoiceAction(Action):
    def __init__(self, n):
        self.ids = list(range(1, n + 1))
        super().__init__(
            command=f'hotbar',
            space=spaces.Discrete(n + 1))

    def xml_template(self) -> str:
        return str("""<InventoryCommands>
            <ModifierList type="allow-list">
            {% for  i in ids %}
            <command>hotbar.{{ i }}</command>
            {% endfor %}
            </ModifierList>
            </InventoryCommands>
            """)

    def to_hero(self, x):
        if x == 0:
            return ""
        return f"hotbar.{x} 1\nhotbar.{x} 0"


class CameraAction(Action):
    def xml_template(self) -> str:
        return str("<CameraCommands/>")

    def to_string(self):
        return 'camera'

    def __init__(self):
        self._command = 'camera'
        super().__init__(self.command, spaces.Box(low=-180., high=180, shape=[2], dtype=np.float32))


class FakeResetAction(Action):
    def xml_template(self) -> str:
        return str("<FakeResetCommand/>")

    def __init__(self):
        super().__init__('fake_reset', spaces.Discrete(2))


class ContinuousNavigationActions(Action):
    def xml_template(self) -> str:
        return str("""<AbsoluteMovementCommands>
            <ModifierList type="allow-list">
            {% for command in commands %}
            <command>{{ command }}</command>
            {% endfor %}
            </ModifierList>
            </AbsoluteMovementCommands>
            """
        )

    def __init__(self, position, ground_level, build_zone):
        self.commands = [
            'tp', #'setPitch', 'setYaw'
        ]
        self.ground_level = ground_level
        self.pos = np.array(list(position)).astype(np.float32)
        if build_zone is not None:
            self.bz1, self.bz2 = build_zone
            self.bz1 = np.array(self.bz1)
            self.bz2 = np.array(self.bz2)
        else:
            self.bz1, self.bz2 = None, None
        super().__init__('navigation', spaces.Dict({
            'move_x': spaces.Box(low=-1., high=1., shape=()),
            'move_y': spaces.Box(low=-1., high=1., shape=()),
            'move_z': spaces.Box(low=-1., high=1., shape=()),
        }))

    def to_hero(self, x):
        cmd = np.array([x[f'move_{k}'] for k in ['x', 'y', 'z']])
        self.pos += cmd
        if self.bz1 is not None:
            self.pos = np.maximum(self.pos, self.bz1)
        if self.bz2 is not None:
            self.pos = np.minimum(self.pos, self.bz2)
        self.pos[1] = max(self.pos[1], self.ground_level)
        coord = ' '.join(map(str, self.pos.round(4).tolist()))
        cmd = f'tp {coord}'
        return cmd

class DiscreteNavigationActions(Action):
    def xml_template(self) -> str:
        return str("""<DiscreteMovementCommands>
            <ModifierList type="allow-list">
            {% for command in commands %}
            <command>{{ command }}</command>
            {% endfor %}
            </ModifierList>
            </DiscreteMovementCommands>
            """
        )

    def __init__(self, movement=True, camera=True, placement=True, custom_name=None):
        self.commands = []
        if movement:
            self.commands += ['move', 'jumpmove', 'strafe', 'jumpstrafe']
        if camera:
            self.commands += ['turn', 'look']
        if placement:
            self.commands += ['attack', 'use']
        self.angle = 0
        space = {}
        if movement:
            space.update({
                'move': spaces.Discrete(3),
                'strafe': spaces.Discrete(3),
                'jump': spaces.Discrete(2),
            })
        if camera:
            space.update({
                "turn": spaces.Discrete(3),
                "look": spaces.Discrete(3),
            })
        if placement:
            space.update({
                "attack": spaces.Discrete(2),
                "use": spaces.Discrete(2)
            })
        space = spaces.Dict(space)
        name = 'navigation' if custom_name is None else custom_name
        super().__init__(name, space)

    def _is_look_command(self, action):
        return 'look' in action and action['look'] != 0
    
    def to_hero(self, x: Union[int, np.int]):
        """
        Returns a command string for the multi command action.
        :param x:
        :return:
        """
        action_list = []
        for movement in ['move', 'strafe', 'turn', 'look']:
            if movement in x and x[movement] != 0:
                way = 1 if x[movement] == 2 else -1
                jmp = ''
                if x['jump'] == 1 and movement in ['move', 'strafe']:
                    jmp = 'jump'
                action_list.append(f'{jmp}{movement} {way}')
        if 'move' in x and x['move'] == 0 and 'strafe' in x and x['strafe'] == 0:
            if 'jump' in x and x['jump'] == 1:
                action_list.append('jump 1')
        for act in ['attack', 'use']:
            if act in x and x[act] != 0:
                action_list.append(f'{act} 1')
        cmd = '\n'.join(action_list)
        if self._is_look_command(x):
            delta = 1 if x[movement] == 2 else -1
            if (self.angle ==  2 and delta ==  1
             or self.angle == -2 and delta == -1):
                # each look 1/-1 action turns agent's camera upwards or downwards
                # by 45 degrees.
                # due to hacky way of DiscreteMovementCommands implementation
                # this doesn't have upper or lower bounds and agent can therefore turn
                # upside down. By this we disallow agent to "roll"
                cmd = '' 
            else:
                self.angle += delta
        return cmd


class HotBarObservation(handlers.FlatInventoryObservation):
    def to_string(self):
        return 'inventory'

    def xml_template(self) -> str:
        return str("<ObservationFromHotBar/>")

    def __init__(self, count):
        handlers.TranslationHandler.__init__(self, spaces.Box(low=0, high=20, shape=(count,)))
        self.num_items = count

    def from_hero(self, info):
        items = self.space.no_op()
        for i in range(self.num_items):
            items[i] = info[f'Hotbar_{i}_size']
        return items


class AgentPosObservation(handlers.TranslationHandlerGroup):
    def xml_template(self) -> str:
        return str("""<ObservationFromFullStats/>""")

    def to_string(self) -> str:
        return "agentPos"

    def __init__(self):
        super().__init__(
            handlers=[
                loc_obs._XPositionObservation(),
                loc_obs._YPositionObservation(),
                loc_obs._ZPositionObservation(),
                loc_obs._PitchObservation(),
                loc_obs._YawObservation(),
            ]
        )
        self.space = spaces.Box(
            low=np.array([-100, 0, -100, -180, -180]),
            high=np.array([100, 100, 100, 180, 180]),
            shape=(5,)
        )

    def from_hero(self, x):
        obs = super().from_hero(x)
        return np.array([obs['xpos'], obs['ypos'] - GROUND_LEVEL - 1, obs['zpos'],
                         obs['pitch'], obs['yaw']])


class String(gym.spaces.Space):
    def noop(self, batch_shape=()):
        return ''
    
    def sample(self, bdim=None):
        return ''
    
    def contains(self, x):
        return x == ''


class ChatObservation(handlers.TranslationHandler):
    def to_string(self) -> str:
        return "chat"

    def xml_template(self) -> str:
        return ""

    def __init__(self, monitor):
        self.monitor = monitor
        super().__init__(space=String())

    def __repr__(self):
        return "String()"
    
    def from_hero(self, x): 
        return self.monitor.current_task.chat


class GridObservation(handlers.TranslationHandler):
    def to_string(self) -> str:
        return "grid"

    def xml_template(self) -> str:
        return str(
            """<ObservationFromGrid>
            <Grid absoluteCoords="true" name="{{ name }}">
            <min x="{{ min_x }}" y="{{ min_y }}" z="{{ min_z }}"/>
            <max x="{{ max_x }}" y="{{ max_y }}" z="{{ max_z }}"/>
            </Grid>
            </ObservationFromGrid>
            """
        )

    def __init__(self, grid_name, min_x, min_y, min_z, max_x, max_y, max_z):
        super().__init__(space=spaces.Box(
            low=0, high=6, shape=BUILD_ZONE_SIZE
            )
        )
        self.name = grid_name
        self.min_x = min_x
        self.min_y = min_y
        self.min_z = min_z
        self.max_x = max_x
        self.max_y = max_y
        self.max_z = max_z

    def from_hero(self, x):
        blocks = x[self.name]
        c = Counter()
        c.update(blocks)
        logger.debug(f'grid obs counts: {c}')
        blocks_id = np.array([block_short2id.get(block, -1) for block in blocks])
        if (blocks_id == -1).any().item():
            logger.warning(f'Wrong block type! grid obs counts: {c}')
        return blocks_id.reshape(*BUILD_ZONE_SIZE)


class TargetGridMonitor(handlers.TranslationHandler):
    def to_string(self):
        return "target_grid"

    def __init__(self, grid_monitor):
        self.grid_monitor = grid_monitor
        super().__init__(space=spaces.Box(low=0, high=1, shape=BUILD_ZONE_SIZE))

    def from_hero(self, x):
        return self.grid_monitor.current_task.target_grid.copy()


class GridIntersectionMonitor(handlers.TranslationHandler):
    def to_string(self):
        return "task"

    def __init__(
            self, grid_name, wrong_placement_reward_scale=1,
            right_placement_reward_scale=2
        ):
        self.wrong_scale = wrong_placement_reward_scale
        self.right_scale = right_placement_reward_scale
        self.grid_name = grid_name
        self.current_task = None
        self.tasks = TaskSet(preset='one_task', task_id='C8')
        self.prev_grid_size = 0
        self.max_int = 0
        super().__init__(space=spaces.Box(low=-2, high=2, shape=()))

    def reset(self):
        self.current_task = self.tasks.sample()
        self.prev_grid_size = 0
        self.max_int = 0

    def set_task(self, task_id):
        self.current_task = self.tasks.set_task(task_id)
        self.prev_grid_size = 0
        self.max_int = 0

    def from_hero(self, x):
        blocks = x[self.grid_name]
        blocks_id = np.array([block_short2id.get(block, -1) for block in blocks])
        grid = blocks_id.reshape(*BUILD_ZONE_SIZE)
        grid_size = (grid != 0).sum().item()
        wrong_placement = (self.prev_grid_size - grid_size) * self.wrong_scale
        max_int = self.current_task.maximal_intersection(grid) if wrong_placement != 0 else self.max_int
        done = max_int == self.current_task.target_size
        self.prev_grid_size = grid_size
        right_placement = (max_int - self.max_int) * self.right_scale
        self.max_int = max_int
        if right_placement == 0:
            reward = wrong_placement
        else:
            reward = right_placement
        return {'reward': reward, 'done': done}
