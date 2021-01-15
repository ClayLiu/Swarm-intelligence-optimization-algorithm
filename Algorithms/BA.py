import numpy as np 
from math import exp
from tqdm import tqdm
from random import random
from Common.constants import *
from Common.SwarmIntelligenceOptimizationAlgorithm import baseIndividual, baseSIOA, IterationResult

class Bat(baseIndividual):
    def __init__(self, search_space : np.ndarray, f_bound = (0, 1)):
        super(Bat, self).__init__(search_space)
        
        self.velocity = np.random.sample(self.x_dim) * 2 - 1    # v_i in (-1, 1)
        
        self.A = 1.0    # A == 1 also can be used
        self.r = random()     # 脉冲率 [0, 1]       
        self.r_zero = self.r

        self.min_f = f_bound[Bounds.lower]
        self.max_f = f_bound[Bounds.upper]
        self.max_min_f = self.max_f - self.min_f

        self.position_new = self.position
    
    def __new_frequency__(self) -> float:
        return random() * self.max_min_f + self.min_f

    def update_global(self, best_position : np.ndarray):
        self.velocity += (self.position - best_position) * self.__new_frequency__()
        self.position_new = self.velocity + self.position

    def search_local(self, ave_A : float) -> np.ndarray:
        eps = random() * 2 - 1
        self.position_new = self.position + eps * ave_A
    
    def update_A_and_r(self, alpha : float, gamma : float, t : int):
        self.A *= alpha
        self.r = self.r_zero * (1 - exp(- gamma * t))

class BatSwarm(baseSIOA):
    individual_class_build_up_func = Bat

    def __init__(self, objective_func, bat_num, search_space, constraint_func = None, alpha = 0.9, gamma = 0.9):
        super(BatSwarm, self).__init__(objective_func, bat_num, search_space, constraint_func)

        self.alpha = alpha
        self.gamma = gamma

    def __build_up_swarm__(self):
        super(BatSwarm, self).__build_up_swarm__()
        self.bat_swarm = self.individual_swarm
        self.bat_num = self.individuals_num

    def __position_bound_check__(self, position : np.ndarray) -> bool:
        return np.all(position < self.search_space[Bounds.upper]) and \
            np.all(position > self.search_space[Bounds.lower])

    def __get_next_generation__(self, best_position : np.ndarray, t : int, best_bat_index : int):
        A_sum = 0.0
        for single_bat in self.bat_swarm:
            A_sum += single_bat.A
        current_ave_A = A_sum / self.bat_num

        for i, single_bat in enumerate(self.bat_swarm):
            subject_pass = False
            # 产生新解
            single_bat.update_global(best_position)

            if random() > single_bat.r:
                single_bat.search_local(current_ave_A)

            if random() < single_bat.A:
                # 判断是否符合约束条件
                if self.constraint_func:
                    subject_pass = self.constraint_func(single_bat.position_new) and self.__position_bound_check__(single_bat.position_new)
                else:
                    subject_pass = self.__position_bound_check__(single_bat.position_new)
                
                # 新解满足约束条件
                if subject_pass:
                    new_fitness = self.objective_func(single_bat.position_new)
                    if new_fitness < self.fitness[i]:
                        self.fitness[i] = new_fitness
                        single_bat.position = single_bat.position_new
                        
                        if new_fitness < self.fitness[best_bat_index]:
                            best_bat_index = i
                            best_position = self.bat_swarm[best_bat_index].position.copy()
                        
                        single_bat.update_A_and_r(self.alpha, self.gamma, t)    


    def iteration(self, iter_num : int, if_show_process = True) -> IterationResult:
        best_fitness_value = []
        
        self.get_fitness()
        best_bat_index = np.argmin(self.fitness)
        best_fitness_value.append(self.fitness[best_bat_index])
        best_position = self.bat_swarm[best_bat_index].position.copy()

        iterator = tqdm(range(1, iter_num + 1)) if if_show_process else range(1, iter_num + 1)   # 根据 if_show_process 选择迭代器

        for t in iterator:
            self.__get_next_generation__(best_position, t, best_bat_index)
            
            self.get_fitness()
            best_bat_index = np.argmin(self.fitness)
            best_fitness_value.append(self.fitness[best_bat_index])
            best_position = self.bat_swarm[best_bat_index].position.copy()

        return IterationResult(
                {
                    'best_position' : best_position,
                    'best_fitness' : self.fitness[best_bat_index],
                    'best_fitness_value_history' : best_fitness_value
                }
            )