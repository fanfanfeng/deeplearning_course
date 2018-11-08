# create by fanfan on 2018/11/8 0008
import random
import matplotlib.pyplot as plt
from reinforcement_learning.algorithmdog.article_five import model_free
from reinforcement_learning.algorithmdog.article_five import mdp
from reinforcement_learning.algorithmdog.article_five import  evaluate
from reinforcement_learning.algorithmdog.article_five import policy


if __name__ == '__main__':
    plt.figure(figsize=(6,3))

    grid_id = mdp.Grid_Mdp_Id()
    policy_id = policy.Policy(grid_id,epsilon=0.5)
    evaler_id = evaluate.Evaler(grid_id)

    policy, y = model_free.mc(grid_id, policy_id, evaler_id, num_iter1=10000, alpha=0.01)
    plt.plot(y, "-", label="mc id feature")

    policy, y = model_free.sarsa(grid_id, policy_id, evaler_id, num_iter1=10000, alpha=0.01);
    plt.plot(y, "--", label="sarsa id feature")

    policy, y = model_free.qlearning(grid_id, policy_id, evaler_id, num_iter1=10000, alpha=0.01)
    plt.plot(y, "-.", label="qlearning id feature")

    plt.xlabel("number of iterations")
    plt.ylabel("square errors")
    plt.legend()
    plt.show()