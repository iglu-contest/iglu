IGLU Environments
=================


.. _MineRL: http://minerl.io


IGLUSilentBuilder-v0
--------------------

This environment is part of Silent Builder task of IGLU competition. 
The agent spawns at the center of building zone which is a `11x9x11`
cuboid above blocks which are marked white. Each step the agent gets a pov image,
an inventory state, a position, and the goal information which is described
below. The agent can navigate inside the building zone, select block stack
from the inventory and place/break blocks. The goal of the agent is to build 
the target structure using only the text of the conversation betwen human 
architect and builder taken from the dataset.

Observation space
*****************

Observation space of silent builder consisits of six components

.. code-block:: python
   
   Dict({
       "pov": Box(low=0, high=255, shape=(64, 64, 3)),
       "inventory": Box(low=0, high=20, shape=(6,)),
       "agentPos": Box(low= [-5, 0, -5, 0,  -90], 
                       high=[ 5, 8,  5, 360, 90],
                       shape=(5,)),
       "grid": Box(low=-1, high=5, shape=(9, 11, 11)),
       "compass": Dict({"angle": Box(low=-180.0, high=180.0, shape=())}),
       "chat": String()
   })

First, ``"pov"`` is a ``64x64`` RGB first-person view image of the agent.
In ``"inventory"`` there are stack counts for each of six block stacks: 
`blue, yellow, green, orange, purple, red`. At the start of the episode
`blue` stack is active. The ``"agentPos"`` component is described by `5`
numbers which are `x, y, z` coordinates and `pitch, yaw` angles.
``"grid"`` observation contains block ids of voxel grid captured from the building zone.
Id `-1` coorresponds to `air` block and the rest of them are ordered as in the 
``"inventory"`` observation. ``"compass"`` component is provided since there is no information
about the dlobal direction inside the images (the building zone looks the same at each direction). 
Finally, ``"chat"`` represents the conversation between
the architect and the builder acquired from human-human interation which coorresponds
to the current task.

Additionally, the agent has access to target structure of the current task. It is 
stored inside ``info`` dictionary by ``'target_grid'`` key. The representation is the same as 
for ``"grid"`` observation component.

.. warning::

    This observation space will not be used for evaluation in the Silent Builder task
    of the IGLU competition. For evaluation environment see ``IGLUSilentBuilderVisual-v0``

Action space
************

The ``IGLUSilentBuilder-v0`` environment can be customized with three different action spaces. 

Human-level actions: 

.. code-block:: python
   
   Dict({
       "forward": Discrete(2),
       "back": Discrete(2),
       "left": Discrete(2),
       "right": Discrete(2),
       "jump": Discrete(2),
       "attack": Discrete(2),
       "use": Discrete(2),
       "camera": Box(low=-180.0, high=180.0, shape=(2,)),
       "hotbar": Discrete(7),
   })

This action space is the same as that in MineRL_ competition environments except there 
are ``"hotbar"`` selection commands added.

Discrete coordinate actions:

.. code-block:: python
   
   Dict({
       "move": Discrete(3),
       "strafe": Discrete(3),
       "jump": Discrete(2),
       "attack": Discrete(2),
       "use": Discrete(2),
       "camera": Box(low=-180.0, high=180.0, shape=(2,)),
       "hotbar": Discrete(7),
   })

Following these actions, the agent would move over discrete positions
coorresponding to centers of blocks.
For navigation commands (``"move"``, ``"strafe"``), there are 3 options
which coorrespond to no-op, forward, and backward movement (no-op, right, and left in 
case of ``"strafe"``). If ``"jump"`` **action is non-zero alongsize the movement action,
the jump would occur simultaneously with movement (as otherwise the agent would be unable 
to jump upstairs).** Take this into account when designing your action space discretization.

Note that states are changed correspondingly immidiately after applying each of these actions.  

Continuous movement actions:

.. code-block:: python
   
   Dict({
       "move_x": Box(low=-1, high=1, shape=()),
       "move_y": Box(low=-1, high=1, shape=()),
       "move_z": Box(low=-1, high=1, shape=()),
       "camera": Box(low=-180.0, high=180.0, shape=(2,)),
       "attack": Discrete(2),
       "use": Discrete(2),
       "hotbar": Discrete(7),
   })

This action space allows agent to fly freely inside the building zone without 
collisions (except with the ground and invisible walls surrounding the building zone). The rest components of the action space 
are the same as in the previous two spaces.

Note that due to how Minecraft processes that kind of events, states are changed with the delay of 2-4 actions. 

To select a proper action space, one can simply pass the corresponding argument
to the environment constructor:

.. code-block:: python

   # For human level actions
   env = gym.make('IGLUSilentBuilder-v0', action_space='human-level')
   # For discrete coordinates movement
   env = gym.make('IGLUSilentBuilder-v0', action_space='discrete')
   # For continuous coordinates movement
   env = gym.make('IGLUSilentBuilder-v0', action_space='continuous')

The default value is ``'human-level'``.


IGLUSilentBuilderVisual-v0
--------------------------

This environment will be used during the evaluation of the solutions
to Silent Builder task of IGLU competition. It provides a reduced observation
space and the same actions.

Observation space
*****************

Observation space of visual silent builder consisits of four components

.. code-block:: python
   
   Dict({
       "pov": Box(low=0, high=255, shape=(64, 64, 3)),
       "inventory": Box(low=0, high=20, shape=(6,)),
       "compass": Dict({"angle": Box(low=-180.0, high=180.0, shape=())}),
       "chat": String()
   })

Each of them was described in the previous section.

Action space
************

In this environment there is again a freedom to select any action space you want.

.. code-block:: python

   # For human level actions
   env = gym.make('IGLUSilentBuilderVisual-v0', action_space='human-level')
   # For discrete coordinates movement
   env = gym.make('IGLUSilentBuilderVisual-v0', action_space='discrete')
   # For continuous coordinates movement
   env = gym.make('IGLUSilentBuilderVisual-v0', action_space='continuous')

The default value is ``'human-level'``.


Reward calculation
------------------


TODO