# create by fanfan on 2018/11/2 0002
import numpy as np
import random
random.seed(0)
from  reinforcement_learning.algorithmdog.article_one import mdp

def random_pi():
    '''
    返回一个随机action
    :return: 
    '''
    actions = ['n','w','e','s']
    r = int(random.random() *4)
    return  actions[r]

def compute_random_pi_state_values():
    value = [ 0.0 for r in range(9)]
    num = 10000

    for k in range(1,num):
        for i in range(1,6):
            mdp_obj = mdp.Mdp()
            state = i
            is_terminal = False
            gamma = 1.0
            v = 0.0
            while False == is_terminal:
                action = random_pi()
                is_terminal ,state,reward = mdp_obj.transform(state,action)
                v = gamma * reward
                gamma *= 0.5

            value[i]= (value[i] * (k-1) + v) / k

        if k % 100 == 0:
            print(value)
    print(value)

compute_random_pi_state_values()
