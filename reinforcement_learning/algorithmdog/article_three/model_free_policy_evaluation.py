# create by fanfan on 2018/11/6 0006
from reinforcement_learning.algorithmdog.article_three import mdp
import random


grid = mdp.Mdp()
states = grid.getStates()
actions = grid.getActions()
gamma = grid.getGamma()

def mc(gamma,state_sample,action_sample,reward_sample):
    vfunc = {}
    nfunc = {}
    for s in states:
        vfunc[s] = 0.0
        nfunc[s] = 0.0

    for iter_ in range(len(state_sample)):
        G = 0.0
        for step in range(len(state_sample[iter_]) -1,-1,-1):
            G *= gamma
            G += reward_sample[iter_][step]

        for step in range(len(state_sample[iter_])):
            state_ = state_sample[iter_][step]
            vfunc[state_] += G
            nfunc[state_] += 1
            G -= reward_sample[iter_][step]
            G /= gamma

    for s in states:
        if nfunc[s] > 0.00001:
            vfunc[s] /= nfunc[s]

    print('mc')
    print(vfunc)
    return vfunc


def td(alpha,gamma,state_sample,action_sample,reward_sample):
    vfunc = {}
    for s in states:
        vfunc[s] = random.random()

    for iter_ in range(len(state_sample)):
        for step in range(len(state_sample[iter_])):
            state = state_sample[iter_][step]
            reward = reward_sample[iter_][step]
            if len(state_sample[iter_]) - 1 > step:
                state_new = state_sample[iter_][step] + 1
                next_v = vfunc[state_new]
            else:
                next_v = 0.0


            vfunc[state] = vfunc[state] + alpha*(reward + gamma*next_v - vfunc[state])


    print('td')
    print(vfunc)
    return vfunc

if __name__ == '__main__':
    s,a,r = grid.gen_randompi_sample(100)
    mc(0.5,s,a,r)
    td(0.15,0.5,s,a,r)
