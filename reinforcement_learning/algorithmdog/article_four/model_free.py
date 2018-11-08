# create by fanfan on 2018/11/7 0007
from reinforcement_learning.algorithmdog.article_three.mdp import Mdp
import random
random.seed(0)
import matplotlib.pyplot as plt


grid = Mdp()
states = grid.getStates()
actions = grid.getActions()
gamma = grid.getGamma()

###############   Compute the gaps between current q and the best q ######
best = {}
def read_best():
    f = open('best_qfunc')
    for line in f:
        line = line.strip()
        if len(line) == 0:
            continue
        tokens = line.split(":")
        best[tokens[0]] = float(tokens[1])

def compute_error(qfunc):
    sum_error = 0.0
    for key in qfunc:
        error = qfunc[key] - best[key]
        sum_error += error * error
    return sum_error

def epsilon_greedy(qfunc,state,epsilon):
    action_max = 0
    key = '%d_%s' % (state,actions[0])
    qvalue_max = qfunc[key]
    for i in range(len(actions)):
        key = "%d_%s" % (state,actions[i])
        q_value = qfunc[key]
        if qvalue_max < q_value:
            qvalue_max = q_value
            action_max = i

    ##probability
    pro = [0.0 for i in range(len(actions))]
    pro[action_max] += 1 - epsilon
    for i in range(len(actions)):
        pro[i] += epsilon/len(actions)

    rand_value = random.random()
    s = 0.0
    for i in range(len(actions)):
        s += pro[i]
        if s >= rand_value:
            return actions[i]
    return actions[len(actions) - 1]

def mc(num_iter1,epsilon):
    x = []
    y = []
    n = {}
    qfunc = {}
    for s in states:
        for a in actions:
            qfunc["%d_%s" % (s,a)] = 0.0
            n['%d_%s' % (s,a)] = 0.001

    for step in range(num_iter1):
        x.append(step)
        y.append(compute_error(qfunc))

        s_sample = []
        a_sample = []
        r_sample = []

        state = states[int(random.random() * len(states))]
        terminal = False
        count = 0
        while False == terminal and count < 100:
            action  = epsilon_greedy(qfunc,state,epsilon)
            terminal ,new_state,reward = grid.transform(state,action)
            s_sample.append(state)
            r_sample.append(reward)
            a_sample.append(action)
            state = new_state
            count += 1

        g = 0.0
        for i in range(len(s_sample)-1,-1,-1):
            g *= gamma
            g += r_sample[i]

        for i in range(len(s_sample)):
            key = "%d_%s" % (s_sample[i],a_sample[i])
            n[key] +=1.0
            qfunc[key] = (qfunc[key] * (n[key] -1) +g)/n[key]

            g -= r_sample[i]
            g /= gamma

    plt.plot(x,y,'_',label="mc epsilon=%2.f"% epsilon)
    return qfunc

def sarsa(num_iter1,alpha,epsilon):
    x = []
    y = []
    qfunc = {}
    for s in states:
        for a in actions:
            key = "%d_%s" % (s,a)
            qfunc[key] = 0.0

    for step in range(num_iter1):
        x.append(step)
        y.append(compute_error(qfunc))

        state = states[int(random.random() * len(states))]
        action = actions[int(random.random() * len(actions))]
        terminal = False
        count = 0
        while False == terminal and count <100:
            key = "%d_%s" % (state,action)
            terminal,new_state,reward = grid.transform(state,action)
            new_action = epsilon_greedy(qfunc,new_state,epsilon)
            key_new = "%d_%s" %(new_state,new_action)
            qfunc[key] = qfunc[key] + alpha *(reward + gamma * qfunc[key_new] -qfunc[key])

            state = new_state
            action = new_action
            count += 1

    plt.plot(x,y,'--',label='sarsa alpha = %2.1f epsilon=%2.1f'%(alpha,epsilon))
    return qfunc

def qlearning(num_iter1,alpha,epsilon):
    x = []
    y = []
    qfunc = {}
    for s in states:
        for a in actions:
            key = "%d_%s" % (s,a)
            qfunc[key] = 0.0

    for step in range(num_iter1):
        x.append(step)
        y.append(compute_error(qfunc))

        state = states[int(random.random() * len(states))]
        action = actions[int(random.random() * len(actions))]
        terminal = False
        count = 0

        while False == terminal and count < 100:
            key = '%d_%s' % (state,action)
            terminal,new_state,reward = grid.transform(state,action)
            key_new = ""
            qmax = -1.0
            for action_ in actions:
                if qmax < qfunc["%d_%s" % (new_state,action_)]:
                    qmax = qfunc["%d_%s" % (new_state,action_)]
                    key_new = "%d_%s" %(new_state,action_)

            qfunc[key] = qfunc[key] + alpha*(reward +gamma*qfunc[key_new] - qfunc[key] )
            state = new_state
            action = epsilon_greedy(qfunc,new_state,epsilon)
            count += 1

    plt.plot(x,y,'-.,',label="q alph=%2.1f epsilon=%2.1f" %(alpha,epsilon))
    return qfunc





if __name__ == '__main__':
    read_best()
    mc(num_iter1=6000, epsilon=0.1)
    sarsa(num_iter1=1000,alpha=0.1,epsilon=0.1)
