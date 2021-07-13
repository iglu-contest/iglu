from collections import OrderedDict
from copy import deepcopy
import numpy as np
import gym


class IgluActionWrapper(gym.ActionWrapper):

    def __init__(self, env):
        super().__init__(env)
        self.noop_dict = OrderedDict({
            'attack': np.array(0),
            'back': np.array(0),
            'camera': np.array([0, 0]),
            'forward': np.array(0),
            'hotbar': np.array(0),
            'jump': np.array(0),
            'left': np.array(0),
            'right': np.array(0),
            'use': np.array(0),
        })

        self.actions_list = [
            ('attack', np.array(1)),
            ('back', np.array(1)),

            ('camera', np.array([5.0, 0.0])),
            ('camera', np.array([-5.0, 0.0])),
            ('camera', np.array([0.0, 5.0])),
            ('camera', np.array([0.0, -5.0])),
            ('camera', np.array([0.0, 0.0])), # noop

            ('forward', np.array(1)),
            # 'hotbar', np.array(0),
            # ('hotbar', np.array(1)),
            # 'hotbar', np.array(2),
            # 'hotbar', np.array(3),
            # 'hotbar', np.array(4),
            # 'hotbar', np.array(5),
            ('jump', np.array(1)),
            ('left', np.array(1)),
            ('right', np.array(1)),
            ('use', np.array(1)),
        ]

        self.action_space = gym.spaces.Discrete(len(self.actions_list))

    def action(self, action):
        result = deepcopy(self.noop_dict)
        result.update({self.actions_list[action][0]: self.actions_list[action][1]})
        return result

    def reverse_action(self, action):
        pass

