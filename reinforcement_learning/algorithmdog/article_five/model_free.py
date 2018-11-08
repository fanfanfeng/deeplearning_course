# create by fanfan on 2018/11/7 0007
from reinforcement_learning.algorithmdog.article_three.mdp import Mdp
from reinforcement_learning.algorithmdog.article_five import evaluate
import random
import numpy as np

def update(policy,fea,action,tvalue,alpha):
    pvalue = policy.qfun(fea,action)
    error = pvalue - tvalue
    fea_new = policy.get_fea_vec(fea,action)
    policy.theta -= alpha * error * fea_new

################ Different model free RL learning algorithms #####

def mc(grid,policy,evaler,num_iter1,alpha):
    actions = grid.actions
    gamma = grid.gamma
    y_loss = []
    for i in range(len(policy.theta)):
        policy.theta[i] = 0.1

    for iter1 in range(num_iter1):
        y_loss.append(evaler.eval(policy))

        state_sample = []
        fea_sample = []
        action_sample = []
        reward_sample = []
        fea = grid.start()
        terminal = False
        count = 0
        while False == terminal and count < 100:
            action = policy.epsilon_greedy(fea)
            state_sample.append(grid.current)
            terminal,fea_new,reward = grid.recieve(action)
            fea_sample.append(fea)
            reward_sample.append(reward)
            action_sample.append(action)
            fea = fea_new
            count += 1

        g = 0.0
        for i in range(len(fea_sample) -1,-1,-1):
            g *= gamma
            g += reward_sample[i]

        for i in range(len(fea_sample)):
            update(policy,fea_sample[i],action_sample[i],g,alpha)
            g - reward_sample[i]
            g /= gamma

    return  policy,y_loss

def sarsa(grid,policy,evaler,num_iter1,alpha):
    actions = grid.actions
    gamma = grid.gamma
    y_loss = []
    for i in range(len(policy.theta)):
        policy.theta[i] = 0.1

    for iter1 in range(num_iter1):
        y_loss.append(evaler.eval(policy))
        fea = grid.start()
        action = actions[int(random.random() * len(actions))]
        terminal = False
        count = 0

        while False == terminal and count < 100:
            terminal,fea_new,reward = grid.recieve(action)
            action_new = policy.epsilon_greedy(fea_new)
            update(policy,fea,action,reward+ gamma* policy.qfun(fea_new,action_new),alpha)

            fea = fea_new
            action = action_new
            count += 1
    return policy,y_loss


def qlearning(grid,policy,evaler,num_iter1,alpha):
    actions = grid.actions
    gamma = grid.gamma
    y_loss =[]
    for i in range(len(policy.theta)):
        policy.theta[i] = 0.1


    for iter1 in range(num_iter1):
        y_loss.append(evaler.eval(policy))

        status = grid.start()
        action = actions[int(random.random() * len(actions))]
        terminal = False
        count = 0

        while False == terminal and count < 100:
            terminal,status_new,reward = grid.recieve(action)

            qmax = -1.0
            for action_ in actions:
                pvalue = policy.qfun(status_new,action_)
                if qmax < pvalue:
                    qmax = pvalue

            update( policy,status,action,reward + gamma * qmax,alpha)

            status = status_new
            action = policy.epsilon_greedy(status)
            count += 1
    return policy,y_loss

