# create by fanfan on 2018/11/9 0009
# !/bin/python

import random

random.seed(10)
import matplotlib.pyplot as plt
from reinforcement_learning.algorithmdog.article_six.gradient import *
from reinforcement_learning.algorithmdog.article_six.policy_value import *
from reinforcement_learning.algorithmdog.article_six.mdp import *
from reinforcement_learning.algorithmdog.article_six.evaluate import *

if __name__ == "__main__":
    plt.figure(figsize=(12, 6))

    grid = Grid_Mdp_Id()
    softmaxpolicy = SoftmaxPolicy(grid, epsilon=0.01)
    valuepolicy = ValuePolicy(grid, epsilon=0.01)
    evaler = Evaler(grid)

    softmaxpolicy, y = saras(grid, evaler, softmaxpolicy, valuepolicy, 2000, 0.01)
    plt.plot(y, "-", label="sarsa")

    plt.xlabel("number of iterations")
    plt.ylabel("square errors")
    plt.legend()
    plt.show()