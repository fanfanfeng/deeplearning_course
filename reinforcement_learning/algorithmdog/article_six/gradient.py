# create by fanfan on 2018/11/9 0009
from reinforcement_learning.algorithmdog.article_six.policy_value import *
import random
import numpy as np

def update_valuepolicy(valuepolicy,state_vec,action,tvalue,alpha):
    pvalue = valuepolicy.qfunc(state_vec,action)
    error = pvalue - tvalue
    fea_vec = valuepolicy.get_fea_vec(state_vec,action)
    valuepolicy.theta -= alpha * error * fea_vec


def update_softmaxpolicy(softmaxpolicy,state_vec,action,qvalue,alpha):
    fea_vec = softmaxpolicy.get_fea_vec(state_vec,action)
    prob = softmaxpolicy.pi(state_vec)

    delta_logJ = fea_vec
    for i in range(len(softmaxpolicy.actions)):
        action_new = softmaxpolicy.actions[i]
        fea_vec_new = softmaxpolicy.get_fea_vec(state_vec,action_new)
        delta_logJ -= fea_vec_new * prob[i]
    delta_logJ *= -1.0
    softmaxpolicy.theta -= alpha * delta_logJ * qvalue

################ Different model free RL learning algorithms #####
def mc(grid,softmaxpolicy,num_iter1,alpha):
    actions = grid.actions
    gamma = grid.gamma
    for i in range(len(softmaxpolicy.theta)):
        softmaxpolicy.theta[i] = 0.1

    for iter1 in range(num_iter1):
        fea_sample = []
        action_sample = []
        reward_sample = []
        state_vec = grid.start()
        terminal = False
        count = 0
        while False == terminal and count <100:
            action = softmaxpolicy.take_action(state_vec)
            terminal,state_vec_new,reward = grid.receive(action)
            fea_sample.append(state_vec)
            reward_sample.append(reward)
            action_sample.append(action)
            state_vec = state_vec_new
            count += 1

        g = 0.0
        for i in range(len(fea_sample) -1,-1,-1):
            g *= gamma
            g += reward_sample[i]

        for i in range(len(fea_sample)):
            update_softmaxpolicy(softmaxpolicy,fea_sample[i],action_sample[i],g,alpha)
            g -= reward_sample[i]
            g /= gamma


    return softmaxpolicy

def saras(grid,evaler,softmaxpolicy,valuepolicy,num_iter1,alpha):
    actions = grid.actions
    gamma = grid.gamma
    y_loss = []
    for i in range(len(valuepolicy.theta)):
        valuepolicy.theta[i] = 0.1

    for i in range(len(softmaxpolicy.theta)):
        softmaxpolicy.theta[i] = 0.0


    for iter1 in range(num_iter1):
        y_loss.append(evaler.eval(valuepolicy))
        state_vec = grid.start()
        action = actions[int(random.random() * len(actions))]
        terminal = False
        count = 0
        while False == terminal and count < 100:
            terminal,state_vec_new,reward = grid.receive(action)
            action_new = softmaxpolicy.take_action(state_vec_new)
            update_valuepolicy(valuepolicy,state_vec,action,reward + gamma * valuepolicy.qfunc(state_vec_new,action_new),alpha)
            update_softmaxpolicy(softmaxpolicy,state_vec,action,valuepolicy.qfunc(state_vec,action),alpha)

            state_vec = state_vec_new
            action = action_new
            count += 1

    return  softmaxpolicy,y_loss

