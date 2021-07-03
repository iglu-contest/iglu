Tasking the environment
=======================

Each episode of the environment is parameterized by the task of the agent. 
Each task is represented by string containing the converation between the architect and the builder
and also by the target structure encoded into 3d voxel grid. Initially, the environment loads the set of tasks
from the dataset collected in :cite:p:`narayan-chen-collaborative`. Each ``env.reset`` samples 
a new task from the current task set and makes it active. We provide several types of sets of tasks
to use with the environment. You can control the set task using the corresponding class, for example 
:py:class:`~iglu.tasks.task_set.RandomTasks`, :py:class:`~iglu.tasks.task_set.TaskSet`, or 
:py:class:`~iglu.tasks.task_set.CustomTasks`.


Random tasks
------------

As a bootstrap for learning, we provide the type of tasks which are generated randomly. 
For this kind of tasks set there will be no conversations, so each goal will have an empty
string inside its ``"chat"`` observation.
You can control the difficulty of generated tasks by changing parameters 
of :py:class:`~iglu.tasks.task_set.RandomTasks` class:

.. autoclass:: iglu.tasks.task_set.RandomTasks
   :noindex:

To use random tasks with the environment, you should call ``.update_taskset`` method.

.. code-block:: python

   import gym
   from iglu.tasks import RandomTasks

   env = gym.make('IGLUSilentBuilder-v0')
   env.update_taskset(RandomTasks(max_blocks=3, max_dist=5, num_colors=3))

   for episode in range(10):
       # for each reset call there will be a completely new target
       obs = env.reset()
       done = False

       while not done:
           action = env.action_space.sample()
           obs, reward, done, info = env.step(action)


Custom tasks
------------

:py:class:`~iglu.tasks.task_set.CustomTasks` class provides an ability to load
and use an arbitrary set of tasks. Each task should contain a conversation 
betwen the architect and the builder; 3d numpy array with target coordinates.
There's no need to specify intermediate steps of building the structure, only the
final structure is required.

An example of building simple custom goal with an imagined conversation:

.. code-block:: python

   import gym
   import numpy as np
   from iglu.tasks import CustomTasks

   env = gym.make('IGLUSilentBuilder-v0')
   custom_grid = np.zeros((9, 11, 11)) # (y, x, z)
   custom_grid[:3, 5, 5] = 1 # blue color
   env.update_taskset(CustomTasks([
       ('<Architect> Please, build a stack of three blue blocks somewhere.\n'
        '<Builder> Sure.', 
        custom_grid)
   ]))

Here is the correspondense between colors and ids:

.. exec:: 

   from iglu.const import id2color; print(id2color)


Minecraft dialogue dataset
--------------------------

All tasks are sampled from the dataset collected 
in :cite:p:`narayan-chen-collaborative`. Each task has its unique id of the format
``C<id>`` where ``<id>`` is a number. On creation, the env loads all tasks from the dataset. 


To load new set of tasks into the environment 

.. code-block:: python

   import gym
   from iglu.tasks import TaskSet

   env = gym.make('IGLUSilentBuilder-v0')
   env.update_taskset(TaskSet(preset=['C1', 'C2', 'C3']))

**By default the env loads task with id** ``C8``. To use all available tasks within the 
set, use ``TaskSet.ALL`` list as a preset. 


Here is the full list of available goals:

.. exec:: 

   from iglu.tasks.task_set import TaskSet
   print('\n'.join(f'{k}: {v}' for k, v in TaskSet.ALL.items()))


References
**********

.. bibliography::
