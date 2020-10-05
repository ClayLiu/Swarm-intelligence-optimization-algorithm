import numpy as np 
from Algorithms.PSO import ParticleSwarm
from Algorithms.SSA import SalpSwarm

if __name__ == '__main__':
    def cost(x : np.ndarray) -> float:
        return (x[0]** 2 - 2) ** 2
    
    search_space = np.array([
        [0],
        [2]
    ])
    PSO_solver = ParticleSwarm(cost, 30, search_space)
    PSO_result = PSO_solver.iteration(1000)

    print(PSO_result.best_position)
    print(PSO_result.best_fitness)

    SSA_solver = SalpSwarm(cost, 30, search_space)
    SSA_result = SSA_solver.iteration(1000)

    print(SSA_result.best_position)
    print(SSA_result.best_fitness)