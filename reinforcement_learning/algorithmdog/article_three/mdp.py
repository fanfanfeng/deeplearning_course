# create by fanfan on 2018/11/2 0002
import numpy as np
import random

class Mdp:
    def __init__(self):
        self.states = [1,2,3,4,5,6,7,8]
        self.terminal_states = {}
        self.terminal_states[6] = 1
        self.terminal_states[7] = 1
        self.terminal_states[8] = 1


        self.actions = ['n','e','s','w']
        self.rewards = {}
        self.rewards['1_s'] = -1.0
        self.rewards['3_s'] = 1.0
        self.rewards['5_s'] = -1.0


        self.t = {}
        self.t['1_s'] = 6
        self.t['1_e'] = 2
        self.t['2_w'] = 1
        self.t['2_e'] = 3
        self.t['3_s'] = 7
        self.t['3_w'] = 2
        self.t['3_e'] = 4
        self.t['4_w'] = 3
        self.t['4_e'] = 5
        self.t['5_s'] = 8
        self.t['5_w'] = 4

        self.gamma  = 0.8

    def transform(self,state,action):
        '''
        :param state: 
        :param action: 
        :return: is_terminal,new_state, reward
        '''
        if state in self.terminal_states.keys():
            return True,state,0

        key = "%d_%s" % (state,action)
        if key in self.t:
            next_state = self.t[key]
        else:
            next_state = state

        is_terminal = False
        if next_state in self.terminal_states.keys():
            is_terminal = True

        if key not in self.rewards:
            reward = 0
        else:
            reward = self.rewards[key]

        return is_terminal,next_state,reward

    def getTerminal(self):
        return self.terminal_states

    def getGamma(self):
        return self.gamma

    def getStates(self):
        return self.states

    def getActions(self):
        return self.actions

    def gen_randompi_sample(self,num):
        state_sample = []
        action_sample = []
        reward_sample = []
        for i in range(num):
            s_tmp = []
            a_tmp = []
            r_tmp = []

            random_state = self.states[int(random.random() * len(self.actions))]
            terminal = False
            while False == terminal:
                random_action = self.actions[int(random.random() * len(self.actions))]
                terminal ,state_new,reward = self.transform(random_state,random_action)
                s_tmp.append(random_state)
                r_tmp.append(reward)
                a_tmp.append(random_action)
                random_state = state_new
            state_sample.append(s_tmp)
            reward_sample.append(r_tmp)
            action_sample.append(a_tmp)

        return state_sample,action_sample,reward_sample




