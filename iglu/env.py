import os
if os.environ.get('MINERL_ENABLE_LOG', '') == '1':
    import sys
    import logging
    logging.getLogger('minerl_patched').setLevel(level=logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    logging.getLogger('minerl_patched').addHandler(handler)

import gym
from gym import spaces
from copy import deepcopy as copy
from typing import List
from minerl_patched.herobraine.env_specs.simple_embodiment import SimpleEmbodimentEnvSpec
from minerl_patched.herobraine.hero.mc import MS_PER_STEP, INVERSE_KEYMAP
import minerl_patched.herobraine.hero.handlers as handlers
from minerl_patched.herobraine.hero.spaces import Dict
from minerl_patched.herobraine.hero.handler import Handler
from minerl_patched.env._singleagent import _SingleAgentEnv

from .handlers import AgentPosObservation, FakeResetAction, \
                      HotBarObservation, \
                      GridObservation, \
                      HotBarChoiceAction, \
                      TargetGridMonitor, \
                      CameraAction, \
                      GridIntersectionMonitor, \
                      DiscreteNavigationActions, \
                      ContinuousNavigationActions, \
                      FakeResetAction, \
                      ChatObservation

from .const import GROUND_LEVEL, block_map, id2block
from .tasks import TaskSet, RandomTasks


class IGLUEnv(_SingleAgentEnv):
    def __init__(
            self, *args, max_steps=500, resolution=(64, 64), 
            start_position=(0.5, GROUND_LEVEL + 1, 0.5, 0., -90),
            bound_agent=True, action_space='human-level', **kwargs
        ) -> None:
        super().__init__(*args, **kwargs)
        self.action_space_type = action_space
        self._tasks = TaskSet(preset='one_task', task_id='C8')
        self.max_steps = max_steps
        self.start_position = start_position
        self.resolution = resolution
        self.bound_agent = bound_agent
        self._should_reset_val = True
        self.counter = 0
        kwargs['env_spec'].action_space_type = action_space
        kwargs['env_spec'].resolution = resolution
        kwargs['env_spec'].bound_agent = bound_agent
        kwargs['env_spec'].start_position = start_position
        self.action_space_ = None

    @property
    def action_space(self):
        if self.action_space_ is None:
            self.task.actionables = self.task.create_actionables()
            self.task._action_space = self.task.create_action_space()
            action_space = self.task.action_space
            self.action_space_ = Dict({
                k: v for k, v in action_space.spaces.items()
                if k != 'fake_reset'
            })
            # flatten space
            if self.action_space_type == 'continuous':
                flatten_func = self.task.flatten_continuous_actions
            elif self.action_space_type == 'discrete':
                flatten_func = self.task.flatten_discrete_actions
            elif self.action_space_type == 'human-level':
                flatten_func = lambda a: (a, (lambda x: x))
            self.action_space_, self.unflatten_action = flatten_func(self.action_space_)
            
        return self.action_space_

    @action_space.setter
    def action_space(self, new_space):
        self.action_space_ = new_space
        if 'fake_reset' in self.action_space_.spaces:
            self.action_space_ = Dict({
                k: v for k, v in self.action_space_.spaces.items()
                if k != 'fake_reset'
            })
    def _init_tasks(self):
        self.spec._kwargs['env_spec'].task_monitor.tasks = self._tasks

    @property
    def tasks(self):
        self._init_tasks()
        return self.spec._kwargs['env_spec'].task_monitor.tasks

    @property
    def _should_reset(self):
        return self._should_reset_val

    def should_reset(self, value: bool):
        self._should_reset_val = value

    @property
    def current_task(self):
        self._init_tasks()
        return self.spec._kwargs['env_spec'].task_monitor.current_task

    def update_taskset(self, tasks):
        self._tasks = tasks
        self.spec._kwargs['env_spec'].task_monitor.tasks = tasks

    def set_task(self, task_id):
        self.spec._kwargs['env_spec'].task_monitor.set_task(task_id)

    def reset(self):
        self.counter = 0
        self._init_tasks()
        if self._should_reset or os.environ.get('IGLU_DISABLE_FAKE_RESET', '0') == '1':
            obs = self.real_reset()
        else:
            fake_reset_action = self.action_space.no_op()
            fake_reset_action['fake_reset'] = 1
            obs, _, done, _ = super().step(fake_reset_action)
            # since this action is handled by minecraft server
            # we need some extra noop actions
            for _ in range(3):
                if not done:
                    obs, _, done, _ = super().step(self.action_space.no_op())
                else:
                    return self.real_reset()
            if done:
                return self.real_reset()
        self.spec._kwargs['env_spec'].task_monitor.reset()
        return obs

    def real_reset(self):
        self.counter = 0
        self._init_tasks()
        obs = super().reset()
        self.should_reset(False)
        return obs

    def step(self, action):
        # TODO: copy action
        action = copy(action)
        action['fake_reset'] = 0
        self.counter += 1
        action = self.unflatten_action(action)
        obs, reward, done, info = super().step(action)
        if 'task' not in info: # connection timed out
            reward = 0
            done = True
            self.should_reset(True)
        else:
            reward = info['task']['reward']
            done = done or info['task']['done']
            del info['task']
        if done:
            self.should_reset(True)
        if self.counter == self.max_steps:
            done = True
        return obs, reward, done, info


class IGLUEnvSpec(SimpleEmbodimentEnvSpec):
    ENTRYPOINT = 'iglu.env:IGLUEnv'
    def __init__(
            self, *args, 
            iglu_evaluation=False, resolution=(64, 64), 
            start_position=(0.5, GROUND_LEVEL + 1, 0.5, 0, -90),
            bound_agent=True, ation_space='human-level', **kwargs
        ):
        self.iglu_evaluation = iglu_evaluation
        self.bound_agent = bound_agent
        self.start_position = start_position
        self.action_space_type = ation_space
        self.task_monitor = GridIntersectionMonitor(grid_name='build_zone')
        if iglu_evaluation:
            name = 'IGLUSilentBuilderVisual-v0'
        else:
            name = 'IGLUSilentBuilder-v0'
        super().__init__(name=name, *args, max_episode_steps=30000,
                         resolution=resolution, **kwargs)

    def _entry_point(self, fake: bool) -> str:
        return IGLUEnvSpec.ENTRYPOINT

    def is_from_folder(self, folder: str) -> bool:
        return False

    def create_agent_mode(self):
        return "Creative" if self.action_space_type == 'continuous' else "Survival"

    def create_agent_start(self):
        # TODO: randomize agent initial position here
        x, y, z, pitch, yaw = self.start_position
        return [
            handlers.AgentStartPlacement(x=x, y=y, z=z, pitch=pitch, yaw=yaw),
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
            handlers.ServerQuitWhenAnyAgentFinishes()
        ]

    def create_observables(self) -> List[Handler]:
        if not self.iglu_evaluation:
            return [
                handlers.POVObservation(self.resolution),
                AgentPosObservation(),
                handlers.CompassObservation(),
                HotBarObservation(6),
                ChatObservation(self.task_monitor),
                GridObservation(
                    grid_name='build_zone',
                    min_x=-5, min_y=GROUND_LEVEL + 1, min_z=-5,
                    max_x=5, max_y=GROUND_LEVEL + 9, max_z=5
                ),
            ]
        else:
            return [
                handlers.POVObservation(self.resolution),
                HotBarObservation(6),
                handlers.CompassObservation(),
                ChatObservation(self.task_monitor),
                GridObservation(
                    grid_name='build_zone',
                    min_x=-5, min_y=GROUND_LEVEL + 1, min_z=-5,
                    max_x=5, max_y=GROUND_LEVEL + 9, max_z=5
                )
            ]

    def create_monitors(self):
        self.task_monitor.reset()
        monitors = [
            self.task_monitor,
        ]
        if not self.iglu_evaluation:
            monitors.append(TargetGridMonitor(self.task_monitor))
        return monitors

    def create_actionables(self):
        if self.action_space_type == 'discrete':
            return self.discrete_actions()
        elif self.action_space_type == 'continuous':
            return self.continuous_actions()
        elif self.action_space_type == 'human-level':
            return self.human_level_actions()

    def discrete_actions(self):
        discrete = DiscreteNavigationActions(movement=True, camera=False, placement=True)
        camera = CameraAction()
        discrete.camera_commands = camera
        return [
            discrete,
            camera,
            HotBarChoiceAction(6),
            FakeResetAction(),
        ]
    
    def flatten_discrete_actions(self, action_space):
        sps = action_space.spaces
        new_space = Dict({
            'move': sps['navigation']['move'],
            'strafe': sps['navigation']['strafe'],
            'jump': sps['navigation']['jump'],
            'attack': sps['navigation']['attack'],
            'use': sps['navigation']['use'],
            'camera': sps['camera'],
            'hotbar': sps['hotbar']
        })
        def unflatten(action_sample):
            updated = {
                'navigation': {
                    k: action_sample[k] for k in ['move', 'strafe', 'jump', 'attack', 'use']
                },
                'camera': action_sample['camera'],
                'hotbar': action_sample['hotbar']
            }
            # need to preserve the rest of keys
            for k in ['move', 'strafe', 'jump', 'attack', 'use', 'camera', 'hotbar']:
                del action_sample[k]
            action_sample.update(updated)
            return action_sample
        return new_space, unflatten


    def continuous_actions(self):
        if self.bound_agent:
            build_zone = [(-5, GROUND_LEVEL + 1, -5),
                          (5, GROUND_LEVEL + 9, 5)]
        else:
            build_zone = None
        x, y, z, _, _ = self.start_position
        return [
            ContinuousNavigationActions(
                (x, y, z), ground_level=GROUND_LEVEL + 1,
                build_zone=build_zone
            ),
            CameraAction(),
            HotBarChoiceAction(6),
            DiscreteNavigationActions(movement=False, camera=False, placement=True, custom_name='placing'),
            FakeResetAction(),
        ]

    def flatten_continuous_actions(self, action_space):
        sps = action_space.spaces
        new_space = Dict({
            'move_x': sps['navigation']['move_x'],
            'move_y': sps['navigation']['move_y'],
            'move_z': sps['navigation']['move_z'],
            'camera': sps['camera'],
            'attack': sps['placing']['attack'],
            'use':    sps['placing']['use'],
            'hotbar': sps['hotbar']
        })
        def unflatten(action_sample):
            updated = {
                'navigation': {f'move_{k}': action_sample[f'move_{k}'] for k in ['x', 'y', 'z']},
                'camera': action_sample['camera'],
                'placing': {'attack': action_sample['attack'], 'use': action_sample['use']},
                'hotbar': action_sample['hotbar'],
            }
            # need to preserve the rest of keys
            for k in ['move_x', 'move_y', 'move_z', 'attack', 'use', 'camera', 'hotbar']:
                del action_sample[k]
            action_sample.update(updated)
            return action_sample
        return new_space, unflatten

    def human_level_actions(self):
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
            HotBarChoiceAction(6),
            FakeResetAction(),
        ]

    def determine_success_from_rewards(self, rewards: list) -> bool:
        return sum(rewards) >= self.reward_threshold

    def get_docstring(self):
        pass


env = IGLUEnvSpec(iglu_evaluation=False)
env.register()
eval_env = IGLUEnvSpec(iglu_evaluation=True)
eval_env.register()
