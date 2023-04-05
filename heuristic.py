def heuristic(scorer, bouncer, paddles, ball_pos, ball_vel, gamesCount):
    if len(scorer) != 2 or len(bouncer) != 2 or len(paddles) != 2:
        raise ValueError("Invalid input values. Please ensure scorer, bouncer, and paddles have two elements each.")

    def determine_training_stage(gamesCount):
        if gamesCount < 10:
            return 'start'
        elif gamesCount < 50:
            return 'early'
        elif gamesCount < 150:
            return 'mid'
        else:
            return 'end'

    training_stage = determine_training_stage(gamesCount)

    def player_heuristic(scorer, bouncer, paddle_pos, opponent_paddle_pos, ball_pos, ball_vel, training_stage):
        heuristic_value = 0

        score_reward = 500
        bounce_reward = 100
        paddle_reward_scale = 1
        speed_penalty_scale = 1

        if training_stage == 'start':
            paddle_reward_scale = 3
            speed_penalty_scale = 0
        elif training_stage == 'early':
            paddle_reward_scale = 2
            speed_penalty_scale = 0.5
        elif training_stage == 'mid':
            bounce_reward = 150
        elif training_stage == 'end':
            paddle_reward_scale = 0.5
            speed_penalty_scale = 1.5

        if scorer == 1:
            heuristic_value += score_reward

        if bouncer == 1:
            heuristic_value += bounce_reward

        paddle_to_ball_distance = abs(paddle_pos - ball_pos[1])
        heuristic_value -= paddle_reward_scale * paddle_to_ball_distance

        heuristic_value -= speed_penalty_scale * abs(ball_vel[0])

        heuristic_value += 1000
        return heuristic_value

    heuristic_A = player_heuristic(scorer[0], bouncer[0], paddles[0], paddles[1], ball_pos, ball_vel, training_stage)
    heuristic_B = player_heuristic(scorer[1], bouncer[1], paddles[1], paddles[0], ball_pos, (-ball_vel[0], ball_vel[1]), training_stage)
    print("heuristic_A: ", heuristic_A)
    print("heuristic_B: ", heuristic_B)
    return [heuristic_A, heuristic_B]
