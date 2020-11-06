import numpy as np 
from math import exp
from tqdm import tqdm

from Common.constants import *
from Algorithms.SSA import SalpSwarm, Salp
from Common.SwarmIntelligenceOptimizationAlgorithm import InterationResult

class Salp_with_pbest(Salp):
    def __init__(self, search_space : np.ndarray):
        super(Salp_with_pbest, self).__init__(search_space)
        self.prev_best_position = self.position.copy()
        self.prev_best_fitness = inf
    
    def update_prev_best(self, fitness : float):
        if self.prev_best_fitness > fitness:
            self.prev_best_fitness = fitness
            self.prev_best_position = self.position.copy()

    def stay_prev_best(self):
        self.position += np.random.random_sample(self.x_dim) * (self.prev_best_position - self.position)


class SalpSwarm_with_prev_best(SalpSwarm):
    '''
        含有历史最优的樽海鞘群算法 \n
        樽海鞘群类，要使用该优化算法，调用 iteration 方法即可 \n
        :param objective_func:  目标函数，参数为 x 向量 \n
        :param particle_num:    樽海鞘个数 \n
        :param search_space:    x 在各维度的取值范围，shape = (2, x_dim) \n
        :param constraint_func: 约束条件函数 返回值为 bool 类型\n
        :param head_num:        头樽海鞘的个数，默认为 1 \n
    '''

    individual_class_build_up_func = Salp_with_pbest


    # def c1_formula(self, t : int, T : int) -> float:
    #     change_point = T >> 1
    #     if t < change_point:
    #         return 2 * ((t - 1) / change_point)
    #     else:
    #         return 2 * exp(- (4 * (t - change_point) / (T - change_point)) ** 2)

    def __get_next_generation__(self, c1 : float):
        for i, salp in enumerate(self.salp_swarm):
            if i < self.head_num:
                salp.update_head(self.global_best_position, c1)
                prev_salp = salp
            else:
                salp.update_other(prev_salp)
                prev_salp = salp
            
            salp.stay_prev_best()
            
            salp.bound_check()
            if self.constraint_func:
                while not self.constraint_func(salp.position):
                    salp.refresh(self.search_space)
            
            fitness = self.objective_func(salp.position)
            salp.update_prev_best(fitness)
            self.fitness[i] = fitness

            if self.global_best_fitness > fitness:
                self.global_best_fitness = fitness
                self.global_best_position = salp.position.copy()
                

    def iteration(self, iter_num : int, if_show_process = True) -> InterationResult:
        '''
            樽海鞘群算法的迭代函数 \n
            :param iter_num: 最大迭代次数 \n
            :param if_show_process: 控制是否显示迭代进度，默认为显示 \n
            :return IterationResult:
        '''
        best_fitness_value_history = []
        
        self.get_fitness()
        best_salp_index = np.argmin(self.fitness)

        self.global_best_fitness = self.fitness[best_salp_index]
        self.global_best_position = self.salp_swarm[best_salp_index].position.copy()

        best_fitness_value_history.append(self.global_best_fitness)

        for salp, fitness in zip(self.salp_swarm, self.fitness):
            salp.update_prev_best(fitness)

        interator = tqdm(range(1, iter_num + 1)) if if_show_process else range(1, iter_num + 1)   # 根据 if_show_process 选择迭代器
        for t in interator:
            c1 = self.c1_formula(t, iter_num)
            self.__get_next_generation__(c1)

            best_fitness_value_history.append(self.global_best_fitness)

        return InterationResult(
                {
                    'best_position' : self.global_best_position,
                    'best_fitness' : self.global_best_fitness,
                    'best_fitness_value_history' : best_fitness_value_history
                }
            )
