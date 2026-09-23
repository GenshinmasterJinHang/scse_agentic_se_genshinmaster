def decide_action(sensor_data, goal_direction):
    if sensor_data['forward_clear']:
        return 'FORWARD'
    elif sensor_data['left_clear'] and goal_direction == 'left':
        return 'LEFT'
    elif sensor_data['right_clear'] and goal_direction == 'right':
        return 'RIGHT'
    else:
        return 'STOP'
