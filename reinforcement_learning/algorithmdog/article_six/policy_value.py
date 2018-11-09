# create by fanfan on 2018/11/9 0009
import random
import numpy as np
class SoftmaxPolicy:
    def __init__(self,grid,epsilon):
        self.actions = grid.actions

        grid.start()
        terminal,state_vec,reward = grid.receive(self.actions[0])
        self.theta = np.array([0.0 for _ in range(len(state_vec.tolist()) * len(self.actions))]).T
        self.epslion = epsilon


    def get_fea_vec(self,state_vec,action):
        fea_vec = np.array([0.0 for i in range(len(self.theta))])
        idx = 0
        for i in range(len(self.actions)):
            if action == self.actions[i]:
                idx = i
                break

        for i in range(len(state_vec)):
            fea_vec[i + idx * len(state_vec)] = state_vec[i]

        return fea_vec

    def pi(self,state_vec):
        prob = [0.0 for i in range(len(self.actions))]
        sum1 = 0.0
        for i in range(len(self.actions)):
            fea_vec = self.get_fea_vec(state_vec,self.actions[i])
            prob[i] = np.exp(np.dot(fea_vec,self.theta))
            sum1 += prob[i]

        for i in range(len(self.actions)):
            prob[i] /= sum1
        return prob

    def take_action(self,state_vec):
        prob = self.pi(state_vec)

        r = random.random()
        s = 0.0
        for i in range(len(self.actions)):
            s += prob[i]
            if s >= r :
                return self.actions[i]

        return self.actions[len(self.actions) - 1]


class ValuePolicy:
    def __init__(self,grid,epsilon):
        self.actions = grid.actions
        grid.start()
        terminal,state_vec,reward = grid.receive(self.actions[0])
        self.theta = np.array([0.0 for _ in range(len(state_vec.tolist()) * len(self.actions))]).T
        self.epslion = epsilon

    def get_fea_vec(self,state_vec,action):
        fea_vec = np.array([0.0 for i in range(len(self.theta))])
        idx = 0
        for i in range(len(self.actions)):
            if action == self.actions[i]:
                idx = i
                break

        for i in range(len(state_vec)):
            fea_vec[i + idx * len(state_vec)] = state_vec[i]

        return fea_vec

    def qfunc(self,state_vec,action):
        fea_vec = self.get_fea_vec(state_vec,action)
        return np.dot(fea_vec,self.theta)

    def epsilon_greedy(self,state_vec):
        epslion = self.epslion
        action_max = 0
        qmax =  self.qfunc(state_vec,self.actions[0])
        for i in range(len(self.actions)):
            action = self.actions[i]
            q_temp = self.qfunc(state_vec,action)
            if qmax < q_temp:
                qmax = q_temp
                action_max = i

        pro = [0.0 for i in range(len(self.actions))]
        pro[action_max] += 1 - epslion

        for i in range(len(self.actions)):
            pro[i] += epslion / len(self.actions)


        rand_num = random.random()
        s = 0.0
        for i in range(len(self.actions)):
            s += pro[i]
            if s >= rand_num:
                return self.actions[i]

        return self.actions[len(self.actions) - 1]







