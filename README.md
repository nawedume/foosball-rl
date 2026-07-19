# Foosball simulator

## Description

A simulator and RL policy that learns to play foosball. Developed in MuJoCo Warp, leveraging RSL RL for the policy training.

## Setup

To test the model.
```
uv sync
uv run python src/test.py
```

Or, you can easily test via the MuJoCo built in viewer. 

```
python -m mujoco.viewer --mjcf model.xml
```

The right hand side should show the "Control" dropdown. Here you can adjust the actuators which control the force. "_sj" refers to sliding joints and "_h" refers to hinge joints.
To reset, hit the "Reload" button on the left side.
