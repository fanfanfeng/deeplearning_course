# create by fanfan on 2018/11/7 0007
from reinforcement_learning.algorithmdog.article_three.mdp import  Mdp
import random
import  numpy as np

class Policy:
    def __init__(self,grid,epsilon):
        self.actions = grid.actions
        grid.start()
        terminal ,state,reward = grid.recieve(self.actions[0])
        self.theta = [ 0.0 for _ in range(len(state) * len(self.actions))]
        self.theta = np.array(self.theta)
        self.theta = np.transpose(self.theta)
        self.epsilon = epsilon


    def get_fea_vec(self,fea,action):
        f = np.zeros(len(self.theta))
        index = 0
        for i in range(len(self.actions)):
            if action == self.actions[i]:
                index = i
                break

        for i in range(len(fea)):
            f[i + index * len(fea)] = fea[i]
        return f

    def qfun(self,fea,a):
        f = self.get_fea_vec(fea,a)
        return np.dot(f,self.theta)



    def epsilon_greedy(self,fea):
        epsilon = self.epsilon
        action_max = 0
        qmax = self.qfun(fea,self.actions[0])
        for i in range(len(self.actions)):
            action_ = self.actions[i]
            q =  self.qfun(fea,action_)
            if qmax < q:
                qmax = q
                action_max = i

        pro =  [0.0 for i in range(len(self.actions))]
        pro[action_max] += 1 - epsilon
        for i in range(len(self.actions)):
            pro[i] += epsilon /len(self.actions)

        rand_num = random.random()
        s = 0.0
        for i in range(len(self.actions)):
            s += pro[i]
            if s >= rand_num:
                return  self.actions[i]

        return  self.actions[len(self.actions) - 1]




