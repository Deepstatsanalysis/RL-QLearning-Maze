# Config

import pandas as pd
import numpy as np
import time

ACTIONS     = ['up', 'down', 'left', 'right']
LENGTH      = None
N_STATES    = None
TERMINAL    = None
EPSILON     = None
MAX_EPISODE = None
LAMBDA      = None
ALPHA       = None

# Initial Q-Table

def build_q_table():
    global N_STATES
    global ACTIONS
    table = pd.DataFrame(
        np.zeros((N_STATES, len(ACTIONS))),
        columns=ACTIONS
    )
    print(table)
    return table

# Actor
# - ε-Greedy

def actor(state, q_table):
    state_act = q_table.iloc[state]
    if np.random.uniform() > EPSILON or state_act.all() == 0:
        act = np.random.choice(ACTIONS)
    else:
        act = state_act.argmax()
    return act

# Enviroment Visual

def update_env(state, episode, step):
    view = np.array([['_ '] * LENGTH] * LENGTH)
    view[tuple(TERMINAL)] = '* '
    view[tuple(state)] = 'o '
    interaction = ''
    for v in view:
        interaction += ''.join(v) + '\n'
    message = 'EPISODE: {}, STEP: {}'.format(episode, step) 
    interaction += message
    if state == TERMINAL:
        print(interaction)
        time.sleep(.5)
    else:
        print(interaction)
        time.sleep(.5)

# Enviroment Feedback

def get_env_feedback(state, action):
    reward = 0.
    a, b = state
    if action == 'up':
        a -= 1
        if a < 0:
            a = 0
        next_state = (a, b)
        if next_state == TERMINAL:
            reward = 1.
    elif action == 'down':
        a += 1
        if a >= LENGTH:
            a = LENGTH - 1
        next_state = (a, b)
    elif action == 'left':
        b -= 1
        if b < 0:
            b = 0
        next_state = (a, b)
    elif action == 'right':
        b += 1
        if b >= LENGTH:
            b = LENGTH - 1
        next_state = (a, b)
        if next_state == TERMINAL:
            reward = 1.
    return next_state, reward

# Run Game

def run():
    q_table = build_q_table()
    episode = 0
    while episode < MAX_EPISODE:
        state = (LENGTH - 1, 0)
        step = 0
        update_env(state, episode, step)
        while state != TERMINAL:
            a, b = state
            act = actor(a * LENGTH + b, q_table)
            print(act)
            next_state, reward = get_env_feedback(state, act)
            na, nb = next_state
            q_predict = q_table.ix[a * LENGTH + b, act]
            if next_state != TERMINAL:
                q_target = reward + LAMBDA * q_table.iloc[na * LENGTH + nb].max()
            else:
                q_target = reward
            q_table.ix[a * LENGTH + b, act] += ALPHA * (q_target - q_predict)
            state = next_state
            step += 1
            update_env(state, episode, step)
        print()
        print(q_table)
        print()
        episode += 1
    return q_table

# Main

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    
    parser.add_argument('-l',
                        default='4',
                        dest="LENGTH",
                        help='input the length of the grid')

    parser.add_argument('-i',
                        default='20',
                        dest='ITERATION',
                        help='input the iteration of training')

    args = parser.parse_args()
    
    LENGTH      = int(args.LENGTH)
    N_STATES    = LENGTH * LENGTH
    TERMINAL    = (0, LENGTH - 1)
    EPSILON     = .9
    MAX_EPISODE = int(args.ITERATION)
    LAMBDA      = .9
    ALPHA       = .1

    q_table = run()
