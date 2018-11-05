# create by fanfan on 2018/11/5 0005
import numpy as np
import random
from reinforcement_learning.algorithmdog.article_one.mdp import Mdp

class Policy_Value:
    def __init__(self,grid_mdp):
        self.value = [ 0.0 for _ in range(len(grid_mdp.states) + 1) ]
        self.pi = {}

        for state in grid_mdp.states:
            if state in grid_mdp.terminal_states:
                continue
            self.pi[state] = grid_mdp.actions[0]

    def policy_improve(self,grid_mdp):
        for state in grid_mdp.states:
            if state in grid_mdp.terminal_states:
                continue
            action_real = grid_mdp.actions[0]
            terminal,new_state,reward = grid_mdp.transform(state,action_real)
            value_new = reward + grid_mdp.gamma * self.value[new_state]

            for action in grid_mdp.actions:
                terminal,new_state,reward = grid_mdp.transform(state,action)
                if value_new < reward + grid_mdp.gamma * self.value[new_state]:
                    action_real = action
                    value_new = reward + grid_mdp.gamma * self.value[new_state]

            self.pi[state] = action_real


    def policy_evalute(self,grid_mdp):
        for i in range(1000):
            delta = 0.0
            for state in grid_mdp.states:
                if state in grid_mdp.terminal_states:
                    continue
                action = self.pi[state]
                terminal,new_state,reward = grid_mdp.transform(state,action)
                new_value = reward + grid_mdp.gamma * self.value[new_state]
                delta += abs(self.value[state] - new_value)
                self.value[state] = new_value

            if delta < 1e-6:
                break

    def policy_iterate(self,grid_mdp):
        for i in range(100):
            self.policy_evalute(grid_mdp)
            self.policy_improve(grid_mdp)



if __name__ == '__main__':
    grid_mdp = Mdp()
    policy_value = Policy_Value(grid_mdp)
    policy_value.policy_iterate(grid_mdp)
    print("Value:")
    for i in range(1,6):
        print("%d:%f\t" % (i,policy_value.value[i]),end=",")
    print("policy:")
    for i in range(1,6):
        print('%d->%s\t' % (i,policy_value.pi[i]),end=',')






