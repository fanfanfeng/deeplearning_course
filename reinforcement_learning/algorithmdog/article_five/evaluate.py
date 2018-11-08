# create by fanfan on 2018/11/7 0007
###############  Compute the gaps between current q and the best q ######
class Evaler:
    def __init__(self,grid):
        self.grid = grid
        self.best = {}
        f = open('./eval.data')
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            elements = line.split(":")
            self.best[elements[0]] = float(elements[1])


    def eval(self,policy):
        grid = self.grid
        sum1 = 0.0
        for key in self.best:
            keys = key.split("_")
            state = int(keys[0])
            if state in grid.terminal_states:
                continue
            fea = grid.start(state)
            action = keys[1]
            error = policy.qfun(fea,action) - self.best[key]
            sum1 += error * error

        return sum1