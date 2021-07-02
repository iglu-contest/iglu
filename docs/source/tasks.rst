Tasking the environment
=======================

Each episode of the environment is parameterized by the task of the agent. 
Each task is represented by string containing the converation and target structure
encoded into 3d voxel grid. All tasks are sampled from the dataset collected 
in :cite:p:`narayan-chen-collaborative`. Each task has its unique id of the format
``C*``.  On creation, the env loads all tasks from the dataset. Each ``.reset`` samples 
a new task from the current task set and makes it active. 

To load new set of tasks into the environment 

.. code-block:: python

   from iglu.tasks import TaskSet

   env = gym.make('IGLUSilentBuilder-v0')
   env.update_taskset(TaskSet(preset=['C1', 'C2', 'C3']))


Additionally, we provide a different type of tasks for the agent. These tasks are generated randomly.
Here are main parameters of ``RandomTasks`` class. It can be used as drop-in replacement for ``TaskSet``.

.. autoclass:: iglu.tasks.task_set.RandomTasks
   :noindex:


Here is the full list of available goals:

.. exec:: 

   from iglu.tasks.task_set import TaskSet
   ts = TaskSet(preset=[f'C{j}' for j in range(1, 158)], update_task_dict=True)
   print('\n'.join(f'{k}: {v}' for k, v in TaskSet.ALL.items()))

References
==========

.. bibliography::
