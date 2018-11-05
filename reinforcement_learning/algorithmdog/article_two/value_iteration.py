# create by fanfan on 2018/11/5 0005
import numpy
import random
from reinforcement_learning.algorithmdog.article_one.mdp import Mdp

class Policy_Value:
    def __init__(self,grid_mdp):
        self.value = [0.0 for i in range(len(grid_mdp.states) + 1)]
        self.pi = {}
        for state in grid_mdp.states:
            if state in grid_mdp.terminal_states:
                continue
            self.pi[state] = grid_mdp.actions[0]

    def value_iteration(self,grid_mdp):
        for i in range(1000):
            delta = 0.0
            for state in grid_mdp.states:
                if state in grid_mdp.terminal_states:
                    continue

                action = grid_mdp.actions[0]
                _,state_new,reward = grid_mdp.transform(state,action)
                value_new = reward + grid_mdp.gamma * self.value[state_new]
                for action_new in grid_mdp.actions:
                    _,state_new,reward = grid_mdp.transform(state,action_new)
                    if value_new < reward + grid_mdp.gamma * self.value[state_new]:
                        action = action_new
                        value_new = reward + grid_mdp.gamma * self.value[state_new]

                delta += abs(value_new  - self.value[state])
                self.pi[state] = action
                self.value[state] = value_new

            if delta < 1e-6:
                break

if __name__ == '__main__':
    grid_mdp = Mdp()
    policy_value = Policy_Value(grid_mdp)
    policy_value.value_iteration(grid_mdp)
    print("value:")
    for i in range(1,6):
        print("%d:%f\t" % (i,policy_value.value[i]),end=' ')

    print("\npolicy:")
    for i in range(1,6):
        print("%d->%s\t" % (i,policy_value.pi[i]),end=" ")
    print("\n")


