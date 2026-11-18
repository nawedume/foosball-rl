import warp as wp

@wp.func
def scale_action(value: float, scale: float):
    # Comparisons preserve NaNs, matching torch.clamp's diagnostic behavior.
    if value < -1.0:
        value = -1.0
    elif value > 1.0:
        value = 1.0
    return value * scale

@wp.kernel
def assign_controls(
    actions: wp.array2d[wp.float32],
    opponent_actions: wp.array2d[wp.float32],
    is_blue: wp.array[wp.bool],
    action_scale: float,
    opponent_scale: float,
    null_opponent: bool,
    actions_scaled: bool,
    opponent_scaled: bool,
    controls: wp.array2d[wp.float32],
):
    env, column = wp.tid()
    rod = column % 8
    player = (column < 8) == is_blue[env]
    value = float(0.0)
    if player:
        value = actions[env, rod]
        if not actions_scaled:
            value = scale_action(value, action_scale)
    elif not null_opponent:
        value = opponent_actions[env, rod]
        if not opponent_scaled:
            value = scale_action(value, opponent_scale)
    controls[env, column] = value


@wp.kernel
def gather_step_state(
    xpos: wp.array2d[wp.vec3],
    geom_xpos: wp.array2d[wp.vec3],
    qvel: wp.array2d[wp.float32],
    sensors: wp.array2d[wp.float32],
    blue_geom_ids: wp.array[wp.int32],
    red_geom_ids: wp.array[wp.int32],
    ball_id: int,
    ball_velocity_address: int,
    blue_sensor: int,
    red_sensor: int,
    max_episode_length: int,
    termination_height: float,
    episode_length: wp.array[wp.int64],
    ball_pos: wp.array[wp.vec3],
    ball_vel: wp.array[wp.vec3],
    blue_players: wp.array2d[wp.vec3],
    red_players: wp.array2d[wp.vec3],
    blue_goals: wp.array[wp.bool],
    red_goals: wp.array[wp.bool],
    out_of_bounds: wp.array[wp.bool],
    time_outs: wp.array[wp.bool],
    dones: wp.array[wp.bool],
):
    env = wp.tid()
    position = xpos[env, ball_id]
    ball_pos[env] = position
    ball_vel[env] = wp.vec3(
        qvel[env, ball_velocity_address],
        qvel[env, ball_velocity_address + 1],
        qvel[env, ball_velocity_address + 2],
    )
    for player in range(blue_geom_ids.shape[0]):
        blue_players[env, player] = geom_xpos[env, blue_geom_ids[player]]
        red_players[env, player] = geom_xpos[env, red_geom_ids[player]]
    episode_length[env] += wp.int64(1)
    timeout = episode_length[env] > wp.int64(max_episode_length)
    blue = sensors[env, blue_sensor] > 0.5
    red = sensors[env, red_sensor] > 0.5
    outside = position[2] <= termination_height
    blue_goals[env] = blue
    red_goals[env] = red
    out_of_bounds[env] = outside
    time_outs[env] = timeout
    dones[env] = timeout or blue or red or outside


@wp.kernel
def advance_reset_seed(dones: wp.array[wp.bool], seed: wp.array[wp.int32]):
    # A device-side decision keeps the seed unchanged for an empty reset mask.
    for env in range(dones.shape[0]):
        if dones[env]:
            seed[0] += 1
            break


@wp.kernel
def invert_sides(is_blue: wp.array[wp.bool], is_red: wp.array[wp.bool]):
    env = wp.tid()
    is_red[env] = not is_blue[env]

@wp.kernel
def prepare_resets(
    dones: wp.array[wp.bool],
    seed: wp.array[wp.int32],
    always_blue: bool,
    is_blue: wp.array[wp.bool],
    episode_length: wp.array[wp.int64],
):
    env = wp.tid()
    if dones[env]:
        rng = wp.rand_init(seed[0], env)
        is_blue[env] = always_blue or wp.randf(rng) < 0.5
        episode_length[env] = wp.int64(0)


@wp.kernel
def randomize_resets(
    dones: wp.array[wp.bool],
    seed: wp.array[wp.int32],
    is_blue: wp.array[wp.bool],
    kick_lesson: bool,
    bias_to_blue: bool,
    ball_position_address: int,
    ball_velocity_address: int,
    ball_start_pos: wp.array[wp.vec3],
    ball_line_length: float,
    joint_ranges: wp.array2d[wp.vec2],
    qpos: wp.array2d[wp.float32],
    qvel: wp.array2d[wp.float32],
):
    env = wp.tid()
    if dones[env]:
        # Independent per-world streams avoid compacting the reset mask on the CPU.
        rng = wp.rand_init(seed[0] + 104729, env)
        if kick_lesson:
            line = wp.randu(rng, wp.uint32(0), wp.uint32(ball_start_pos.shape[0]))
            position = ball_start_pos[line]
            position[1] -= wp.randf(rng) * ball_line_length
            direction = float(1.0)
            if is_blue[env]:
                direction = -1.0
            position[0] += wp.randf(rng) * 0.01 * direction
            for component in range(3):
                qpos[env, ball_position_address + component] = position[component]
        else:
            bias = float(0.0)
            if bias_to_blue:
                bias = 1.0
            qvel[env, ball_velocity_address] = wp.randf(rng) + bias
            qvel[env, ball_velocity_address + 1] = -wp.randf(rng) * 2.0
            qpos[env, ball_position_address] += (wp.randf(rng) - 0.5) * 0.8
            qpos[env, ball_position_address + 1] += (wp.randf(rng) - 0.5) * 0.6
        for joint in range(16):
            minimum = joint_ranges[0, joint][0]
            maximum = joint_ranges[0, joint][1]
            qpos[env, joint] = minimum + wp.randf(rng) * (maximum - minimum)

@wp.kernel
def compute_observations(
    qpos: wp.array2d[wp.float32],
    qvel: wp.array2d[wp.float32],
    xpos: wp.array2d[wp.vec3],
    is_red: wp.array[wp.bool],
    blue_goal: wp.array2d[wp.float32],
    red_goal: wp.array2d[wp.float32],
    ball_id: int,
    ball_velocity_address: int,
    observations: wp.array2d[wp.float32],
):
    env, column = wp.tid()
    red = is_red[env]
    if column < 32:
        # Four groups: player positions/velocities, opponent positions/velocities.
        group = column // 8
        rod = column % 8
        offset = 0
        if (group < 2 and red) or (group >= 2 and not red):
            offset = 8
        if group % 2 == 0:
            observations[env, column] = qpos[env, offset + rod]
        else:
            observations[env, column] = qvel[env, offset + rod]
    else:
        component = (column - 32) % 3
        value = float(0.0)
        if column < 35:
            goal = blue_goal[0, component]
            if red:
                goal = red_goal[0, component]
            value = xpos[env, ball_id][component] - goal
        else:
            value = qvel[env, ball_velocity_address + component]
        if red and component < 2:
            value = -value
        observations[env, column] = value

@wp.kernel
def set_random_ball_position_to_player(
    envs: wp.array[wp.int32],
    ball_idx: wp.int32,
    seed: wp.int32,
    step: wp.int32,
    is_blue: wp.array[wp.bool],
    ball_start_pos: wp.array[wp.vec3], # size of number of sites
    ball_start_line_length: wp.float32,
    out_xpos: wp.array2d[wp.float32],
):
    env_idx = envs[wp.tid()]

    num_sites = wp.uint32(ball_start_pos.shape[0])

    rng = wp.rand_init(seed + step, env_idx)
    line_idx = wp.randu(rng, wp.uint32(0), num_sites)
    t = wp.randf(rng)

    # y because thats the axis of the rod
    pos = ball_start_pos[line_idx]
    pos[1] -= t*ball_start_line_length
    # go towards the opposite end (blue aims toward the -x)
    pos[0] += wp.randf(rng) * 0.01 * (-1.0 if is_blue[env_idx] else 1.0)

    out_xpos[env_idx, ball_idx + 0] = pos[0]
    out_xpos[env_idx, ball_idx + 1] = pos[1]
    out_xpos[env_idx, ball_idx + 2] = pos[2]

@wp.func
def compute_alignment_reward(val: wp.float32):
    # always positive, near 0.1m from the value (cloest lateral postion), times 2  to enforce higher reward
    return ( wp.max(0.0, (0.1 - val)) / 0.1 ) * 2.0

@wp.kernel
def compute_rewards(
    is_blue: wp.array[wp.bool],
    blue_goals: wp.array[wp.bool],
    red_goals: wp.array[wp.bool],
    ball_pos: wp.array[wp.vec3],
    ball_vel: wp.array[wp.vec3],
    blue_goal_center: wp.vec3,
    red_goal_center: wp.vec3,
    out_of_bounds: wp.array[wp.bool],

    blue_rod_pos: wp.array[wp.vec3],
    red_rod_pos: wp.array[wp.vec3],
    blue_player_pos: wp.array2d[wp.vec3],
    red_player_pos: wp.array2d[wp.vec3],

    goal_reward_multiplier: wp.float32,
    oob_reward_multiplier: wp.float32,

    # form: [total, r_goal, r_oob, r_dist, r_vel, r_align]
    rewards: wp.array2d[wp.float32],
):
    env = wp.tid()

    if is_blue[env]:
        team_goals, op_goals = (blue_goals, red_goals)
        rod_pos = blue_rod_pos
        player_pos = blue_player_pos
    else:
        team_goals, op_goals = (red_goals, blue_goals)
        rod_pos = red_rod_pos
        player_pos = red_player_pos

    r_goal = 1.5*goal_reward_multiplier*float(op_goals[env]) - goal_reward_multiplier*float(team_goals[env])

    op_goal_center =  red_goal_center if is_blue[env] else blue_goal_center
    dist_v = op_goal_center[0:2] - ball_pos[env][0:2]
    dist = wp.norm_l2(dist_v)
    r_dist = -dist * 4.0

    dist_unit = dist_v / wp.max(dist, 1e-5) #  ensure no divide by 0
    r_vel = max(0.0, wp.dot(ball_vel[env][0:2], dist_unit) * 20.0)

    r_oob = float(out_of_bounds[env]) * oob_reward_multiplier

    bx = ball_pos[env][0]
    by = ball_pos[env][1]

    r_align = 0.0
    vertical_reach = 0.1
    # if its vertical_reach meters close to the rod in the standard order, 4 rods
    if (vertical_reach - wp.abs(bx - rod_pos[0][0])) > 0.0:
        p1, p2, p3 = player_pos[env, 0][1], player_pos[env, 1][1], player_pos[env, 2][1]
        a1 = wp.abs(by - p1)
        a2 = wp.abs(by - p2)
        a3 = wp.abs(by - p3)
        r_align = compute_alignment_reward(wp.min(a1, wp.min(a2, a3)))
    elif (vertical_reach - wp.abs(bx - rod_pos[1][0])) > 0.0:
        p1, p2, p3 = player_pos[env, 3][1], player_pos[env, 4][1], player_pos[env, 5][1]
        a1 = wp.abs(by - p1)
        a2 = wp.abs(by - p2)
        a3 = wp.abs(by - p3)
        r_align = compute_alignment_reward(wp.min(a1, wp.min(a2, a3)))
    elif (vertical_reach - wp.abs(bx - rod_pos[2][0])) > 0.0:
        p1, p2 = player_pos[env, 6][1], player_pos[env, 7][1]
        a1 = wp.abs(by - p1)
        a2 = wp.abs(by - p2)
        r_align = compute_alignment_reward(wp.min(a1, a2))
    elif (vertical_reach - wp.abs(bx - rod_pos[3][0])) > 0.0:
        p1, p2, p3, p4, p5 = \
            player_pos[env, 8][1], player_pos[env, 9][1], player_pos[env, 10][1], player_pos[env, 11][1], player_pos[env, 12][1]
        a1 = wp.abs(by - p1)
        a2 = wp.abs(by - p2)
        a3 = wp.abs(by - p3)
        a4 = wp.abs(by - p4)
        a5 = wp.abs(by - p5)
        r_align = compute_alignment_reward(wp.min(a1, wp.min(a2, min(a3, min(a4, a5)))))

    r_total = r_goal + r_oob + r_dist + r_vel + r_align
    rewards[env, 0] = r_total
    rewards[env, 1] = r_goal
    rewards[env, 2] = r_oob
    rewards[env, 3] = r_dist
    rewards[env, 4] = r_vel
    rewards[env, 5] = r_align
