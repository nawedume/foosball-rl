import warp as wp

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