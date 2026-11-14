# Experiment 1

This experiment looked at the standard reward function with the following terms:

- Ball Distance (negative)
- Velocity towards the target (Min 0)
- Goals
- OOB Penalty
- Alignment

This is standard torque/force control. Trained in 1000 iterations against a null opponent. The ball was randomized using the 
"kicking" training, which spawns the ball on the line of one of the blue rods.

The model still seems to struggle finding alignment, when the ball passes or even stationary, it doesn't properly
map the right force control to the reward of hitting the ball in the right direction (via distance and velocity).

This could be because the ball contact is still sparse and a rollout window of 128 and gamma of 0.995 isn't sufficient to map
later rewards to the action of moving the rod laterally. However, the rapid turning of the rods to kick does indicate that
the model is successfully mapping the action of kicking to a high reward. More often than not, visually, it appears to kick
in the right direction. Although that might be a consequence of randomization intiialization.

Some next experiements to see:

- Train with a chpt opponent, this might make ball hits less sparse since the ball won't get stuck at a red rod nearly as often.
- Position control instead of torque control via implicit actuation, so the model doesn't need to learn the mapping of torque
to postion and velocity.
- Larger critic model, perhaps the model is struggling to determine how good the future rewards can be.