# Learning to race with Reinforcement Learning  

![](https://github.com/meraccos/f1tenth_reinforcement_learning/blob/main/racing.gif)

(watch the full video [here](https://youtu.be/10Nks2jMxBk))   
  
This repository is originally forked from [here](https://github.com/f1tenth/f1tenth_gym). Check for original documentation.  
As the original repo is mainly designed for SLAM purposes, there are significant changes to the environment structure.  
  
The RL content can be found under the folder [work](https://github.com/meraccos/f1tenth_reinforcement_learning/tree/main/work).  
  
I will try to add more documentation, but the file names are quite self-explanatory.  
  
You can start the training by running `train.py`. There are several ways to debug your code, including:

* Running `eval.py` will run several episodes and print out the average episodic reward.
* Running `eval_test.py` will run a trained episode with rendering on.
* Running `keyboard_control.py` will let you drive the car with keyboard.
* Running `map_test.py` will you check whether your centerpoint data matches the map jpg.
* Running `lidar_test.py` and `keyboard_lidar_test` will let you inspect the lidar data.

Feel free to contact me for more details!  
  
You can find the [documentation](https://f1tenth-gym.readthedocs.io/en/latest/) of the environment here.  
  
## Quickstart
We recommend installing the simulation inside a virtualenv. You can install the environment by running:

```bash
virtualenv gym_env
source gym_env/bin/activate
git clone https://github.com/f1tenth/f1tenth_gym.git
cd f1tenth_gym
pip install -e .
```

Then you can run a quick waypoint follow example by:
```bash
cd examples
python3 waypoint_follow.py
```

A Dockerfile is also provided with support for the GUI with nvidia-docker (nvidia GPU required):
```bash
docker build -t f1tenth_gym_container -f Dockerfile .
docker run --gpus all -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix f1tenth_gym_container
````
Then the same example can be ran.

## Known issues
- Library support issues on Windows. You must use Python 3.8 as of 10-2021
- On MacOS Big Sur and above, when rendering is turned on, you might encounter the error:
```
ImportError: Can't find framework /System/Library/Frameworks/OpenGL.framework.
```
You can fix the error by installing a newer version of pyglet:
```bash
$ pip3 install pyglet==1.5.11
```
And you might see an error similar to
```
gym 0.17.3 requires pyglet<=1.5.0,>=1.4.0, but you'll have pyglet 1.5.11 which is incompatible.
```
which could be ignored. The environment should still work without error.

## Citing
If you find this Gym environment useful, please consider citing:

```
@inproceedings{okelly2020f1tenth,
  title={F1TENTH: An Open-source Evaluation Environment for Continuous Control and Reinforcement Learning},
  author={O’Kelly, Matthew and Zheng, Hongrui and Karthik, Dhruv and Mangharam, Rahul},
  booktitle={NeurIPS 2019 Competition and Demonstration Track},
  pages={77--89},
  year={2020},
  organization={PMLR}
}
```
