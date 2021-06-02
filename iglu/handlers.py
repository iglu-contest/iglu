from collections import Counter
import logging
import numpy as np

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
        super().__init__(space=spaces.Box(low=-1, high=1, shape=()))

    def reset(self):
        self.current_task = self.tasks.sample()
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
