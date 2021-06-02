import os
if os.environ.get('MINERL_ENABLE_LOG', '') == '1':
    import sys
    import logging
    logging.getLogger('minerl').setLevel(level=logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    logging.getLogger('minerl').addHandler(handler)

from typing import List
from minerl.herobraine.env_specs.simple_embodiment import SimpleEmbodimentEnvSpec
from minerl.herobraine.hero.mc import MS_PER_STEP, INVERSE_KEYMAP
import minerl.herobraine.hero.handlers as handlers
from minerl.herobraine.hero.handler import Handler
from minerl.env._singleagent import _SingleAgentEnv

from .handlers import AgentPosObservation, \
                      HotBarObservation, \
                      GridObservation, \
                      HotBarChoiceAction, \
                      TargetGridMonitor, \
                      GridIntersectionMonitor

from .const import GROUND_LEVEL, block_map, id2block
from .wrappers import RewardFromInfo
from .tasks import TaskSet, RandomTasks


class IGLUEnv(_SingleAgentEnv):
    def _init_tasks(self):
        if not hasattr(self, '_tasks'):
            self._tasks = TaskSet(preset='one_task', task_id='C8')
        self.spec._kwargs['env_spec'].task_monitor.tasks = self._tasks

    @property
    def tasks(self):
        self._init_tasks()
        return self.spec._kwargs['env_spec'].task_monitor.tasks

    def update_taskset(self, tasks):
        self._tasks = tasks
        self.spec._kwargs['env_spec'].task_monitor.tasks = tasks

    def reset(self):
        obs = super().reset()
        self._init_tasks()
        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)
        if 'task' not in info: # connection timed out
            reward = 0
            done = False
        else:
            reward = info['task']['reward']
            done = done or info['task']['done']
            del info['task']
        return obs, reward, done, info


class IGLUEnvSpec(SimpleEmbodimentEnvSpec):
    ENTRYPOINT = 'iglu.env:IGLUEnv'
    def __init__(self, *args, **kwargs):
        self.task_monitor = GridIntersectionMonitor(grid_name='build_zone')
        super().__init__(name='IGLUSilentBuilder-v0', *args, max_episode_steps=2000,
                         resolution=(64, 64), **kwargs)

    def make(self, **kwargs):
        env = super().make(**kwargs)
        env = RewardFromInfo(env)
        return env

    def _entry_point(self, fake: bool) -> str:
        return IGLUEnvSpec.ENTRYPOINT

    def is_from_folder(self, folder: str) -> bool:
        return folder == 'survivaltreechop'

    def create_agent_start(self):
        # TODO: randomize agent initial position here
        return [
            handlers.AgentStartPlacement(x=-1, y=GROUND_LEVEL + 1, z=0, pitch=0, yaw=-90),
            handlers.InventoryAgentStart({
                i: {'type': v, 'quantity': 20} for i, v in enumerate(block_map.values())
            })
        ]

    def create_agent_handlers(self) -> List[Handler]:
        return []

    def create_server_world_generators(self):
        return [handlers.FlatWorldGenerator(
            force_reset=True, generatorString=f"3;7,{GROUND_LEVEL}*malmomod:iglu_unbreakable_grey_rn;3;biome_1"
        )]

    def create_server_initial_conditions(self):
        return [
            handlers.TimeInitialCondition(
                start_time=6000,
                allow_passage_of_time=False,
            ),
            handlers.WeatherInitialCondition(weather="clear"),
            handlers.SpawningInitialCondition(
                allow_spawning=False
            )
        ]

    def create_server_decorators(self) -> List[Handler]:
        return [
            handlers.DrawingDecorator(
                f'<DrawCuboid type="malmomod:iglu_unbreakable_white_rn" x1="-5" y1="{GROUND_LEVEL}" z1="-5" x2="5" y2="{GROUND_LEVEL}" z2="5"/>' 
            )
        ]

    def create_rewardables(self) -> List[Handler]:
        return []

    def create_server_quit_producers(self):
        return [
            handlers.ServerQuitFromTimeUp(time_limit_ms=
                                          self.max_episode_steps * MS_PER_STEP),
            handlers.ServerQuitWhenAnyAgentFinishes()]

    def create_observables(self) -> List[Handler]:
        return [
            handlers.POVObservation(self.resolution),
            AgentPosObservation(),
            HotBarObservation(6),
            GridObservation(
                grid_name='build_zone',
                min_x=-5, min_y=GROUND_LEVEL + 1, min_z=-5,
                max_x=5, max_y=GROUND_LEVEL + 9, max_z=5
            ),
        ]

    def create_monitors(self):
        self.task_monitor.reset()
        return [
            self.task_monitor, 
            TargetGridMonitor(self.task_monitor)
        ]

    def create_actionables(self):
        SIMPLE_KEYBOARD_ACTION = [
            "forward",
            "back",
            "left",
            "right",
            "jump",
            "attack",
            "use"
        ]
        return [
            handlers.KeybasedCommandAction(k, v) for k, v in INVERSE_KEYMAP.items()
            if k in SIMPLE_KEYBOARD_ACTION
        ] + [
            handlers.CameraAction(),
            HotBarChoiceAction(6)
        ]

    def determine_success_from_rewards(self, rewards: list) -> bool:
        return sum(rewards) >= self.reward_threshold

    def get_docstring(self):
        pass


env = IGLUEnvSpec()
env.register()
