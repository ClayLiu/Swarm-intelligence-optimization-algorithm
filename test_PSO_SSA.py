import numpy as np 
import matplotlib.pyplot as plt 
from Algorithms.PSO import ParticleSwarm
from Algorithms.SSA import SalpSwarm
from Utils import test_functions


if __name__ == '__main__':
    n = 4
    tester = test_functions.Schwefel(n)

    search_space = tester.search_space
    # search_space[1] *= 0 
    cost = tester.func

    PSO_solver = ParticleSwarm(cost, 30, search_space)
    PSO_result = PSO_solver.iteration(10000)

    print(PSO_result.best_position)
    print(PSO_result.best_fitness)
    plt.title('PSO fitness')
    plt.plot(PSO_result.best_fitness_value_history)


    SSA_solver = SalpSwarm(cost, 30, search_space)
    SSA_result = SSA_solver.iteration(10000)

    print(SSA_result.best_position)
    print(SSA_result.best_fitness)
    plt.figure()
    plt.title('SSA fitness')
    plt.plot(SSA_result.best_fitness_value_history)

    plt.show()