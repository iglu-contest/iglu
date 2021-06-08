from collections import Counter
import logging
import numpy as np
from typing import Union

from minerl.herobraine.hero.handlers.agent.action import Action
from minerl.herobraine.hero import spaces
import minerl.herobraine.hero.handlers as handlers
import minerl.herobraine.hero.handlers.agent.observations.location_stats as loc_obs

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
        super().__init__(self.command, spaces.Box(low=-5, high=5, shape=[2], dtype=np.float32))


class AbsoluteNavigationActions(Action):
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

    def __init__(self, position, pitch, yaw, ground_level, build_zone):
        self.commands = [
            'tp', #'setPitch', 'setYaw'
        ]
        self.ground_level = ground_level
        self.pos = np.array(list(position))
        self.pitch = pitch
        self.yaw = yaw
        self.bz1, self.bz2 = build_zone
        self.bz1 = np.array(self.bz1)
        self.bz2 = np.array(self.bz2)
        self.navigation_commands = {
            'movenorth': np.array([0, 0, -1]),
            'movesouth': np.array([0, 0, -1]),
            'moveeast': np.array([1, 0, 0]),
            'movewest': np.array([-1, 0, 0]),
            'up': np.array([0, 1, 0]),
            'down': np.array([0, -1, 0]),
        }
        self.action_map = list(self.navigation_commands.keys())
        self.inverse_map = {
            v: i for i, v in enumerate(self.action_map)
        }
        super().__init__('abs_navigation', spaces.Discrete(
            len(self.action_map)))

    def to_hero(self, x: Union[int, np.int]):
        """
        Returns a command string for the multi command action.
        :param x:
        :return:
        """
        if not isinstance(x, int):
            x = x.item()
        x_id = x
        x = self.action_map[x]
        if x_id < len(self.navigation_commands):
            cmd = self.navigation_commands[x]
            self.pos += cmd
            self.pos = np.maximum(self.pos, self.bz1)
            self.pos = np.minimum(self.pos, self.bz2)
            self.pos[1] = max(self.pos[1], self.ground_level)
            coord = ' '.join(map(str, self.pos.tolist()))
            cmd = f'tp {coord}'
        else:
            cmd = self.camera_commands[x]
            pitch_delta, yaw_delta = cmd
            self.pitch = np.clip(self.pitch + pitch_delta, -90, 90)
            self.yaw = (self.yaw + yaw_delta) % 360
            cmd = f'setPitch {self.pitch}\nsetYaw {self.yaw}'
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

    def __init__(self, movement=True, camera=True, placement=True):
        self.commands = []
        if movement:
            self.commands += ['move', 'jumpmove', 'strafe', 'jumpstrafe']
        if camera:
            self.commands += ['turn', 'look']
        if placement:
            self.commands += ['attack', 'use']
        self.angle = 0
        self.action_map = [
            '', # no-op
        ]
        if movement:
            self.action_map += [
                'move 1',
                'move -1',
                'jumpmove 1',
                'jumpmove -1',
                'strafe 1',
                'strafe -1',
                'jumpstrafe 1',
                'jumpstrafe -1',
            ]
        if camera:
            self.action_map += [
                'turn 1',
                'turn -1',
                'look 1',
                'look -1',    
            ]
        if placement:
            self.action_map = [
                'attack 1',
                'use 1'
            ]
        self.inverse_map = {
            v: i for i, v in enumerate(self.action_map)
        }
        super().__init__('navigation', spaces.Discrete(len(self.action_map)))

    def _is_look_command(self, action_id):
        return 'look' in self.action_map[action_id] 
    
    def to_hero(self, x: Union[int, np.int]):
        """
        Returns a command string for the multi command action.
        :param x:
        :return:
        """
        if not isinstance(x, int):
            x = x.item()
        cmd = self.action_map[x]
        if self._is_look_command(x):
            delta = int(cmd[len('look '):])
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
