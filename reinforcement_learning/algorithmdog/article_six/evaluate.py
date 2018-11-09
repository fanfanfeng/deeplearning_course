# create by fanfan on 2018/11/9 0009
from reinforcement_learning.algorithmdog.article_six.mdp import Grid_Mdp_Id

class Evaler:
    def __init__(self,grid):
        self.grid = grid
        self.best = {}
        f = open('./eval.data')
        for line in f:
            line = line.strip()
            if len(line) == 0 :
                continue
            tokes = line.split(":")
            self.best[tokes[0]] = float(tokes[1])

    def eval(self,value_policy):
        grid = self.grid
        sum_all = 0.0
        for key in self.best:
            keys = key.split("_")
            state = int(keys[0])
            if state in grid.terminal_states:
                continue
            state_vec = grid.start(state)
            action = keys[1]

            error = value_policy.qfunc(state_vec,action) - self.best[key]
            sum_all += error * error
        return sum_all