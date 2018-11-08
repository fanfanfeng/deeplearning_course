# create by fanfan on 2018/11/7 0007
from reinforcement_learning.algorithmdog.article_three.mdp import Mdp
from reinforcement_learning.algorithmdog.article_four import model_free
import random
random.seed(0)
import matplotlib.pyplot as plt

if __name__ == '__main__':
    model_free.read_best()
    plt.figure(figsize=(12,6))

    ############# variance ##################
    from reinforcement_learning.algorithmdog.article_four import model_free

    from reinforcement_learning.algorithmdog.article_four import model_free

    model_free.mc(num_iter1=6000, epsilon=0.2)
    model_free.mc(num_iter1=6000, epsilon=0.2)
    model_free.mc(num_iter1=6000, epsilon=0.2)
    model_free.mc(num_iter1=6000, epsilon=0.2)
    model_free.sarsa(num_iter1=6000, alpha=0.2, epsilon=0.2)
    model_free.sarsa(num_iter1=6000, alpha=0.2, epsilon=0.2)
    model_free.sarsa(num_iter1=6000, alpha=0.2, epsilon=0.2)
    model_free.sarsa(num_iter1=6000, alpha=0.2, epsilon=0.2)
    model_free.qlearning(num_iter1=6000, alpha=0.2, epsilon=0.2)
    model_free.qlearning(num_iter1=6000, alpha=0.2, epsilon=0.2)
    model_free.qlearning(num_iter1=6000, alpha=0.2, epsilon=0.2)
    model_free.qlearning(num_iter1=6000, alpha=0.2, epsilon=0.2)

    plt.xlabel('number of iterations')
    plt.ylabel('square errors')
    plt.legend()
    plt.show()
