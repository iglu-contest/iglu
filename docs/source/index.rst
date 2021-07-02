IGLU: Grounded Language Understanding in Minecraft
==================================================

.. image:: assets/bell.png
  :scale: 20 %
  :alt:
.. image:: assets/L.png
  :scale: 20 %
  :alt:
.. image:: assets/heart.png
  :scale: 20 %
  :alt:
.. image:: assets/Ls.png
  :scale: 20 %
  :alt:

.. _MineRL: http://minerl.io

What is IGLU?
-------------

IGLU is a research project aimed at bridging the gap between reinforcement learning and 
natural language understanding in Minecraft. It provides an RL environment where the goal 
of an agent is to build structures within a dedicated zone. The structures are described
by natural language in the game's chat. 


.. image:: assets/dialog_example.png
  :scale: 60 %
  :alt:

We thank creators of MineRL_, as our codebase depends heavily on their project.

.. toctree::
   :caption: Getting started
   :maxdepth: 1

   installation
   envs
   tasks


Getting started
===============

.. code-block:: python

   import gym
   import iglu

   env = gym.make('IGLUSilentBuilder-v0', max_steps=1000)
   obs = env.reset()
   done = False

   while not done:
       action = env.action_space.sample()
       obs, reward, done, info = env.step(action)
       




Package reference
=================

.. toctree::
   :maxdepth: 2
   :caption: IGLU API reference:

   auto/iglu
   auto/iglu.tasks
   auto/modules


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
