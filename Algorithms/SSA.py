import numpy as np 
from math import exp
from random import random
from tqdm import tqdm

from Common.constants import *
from Common.SwarmIntelligenceOptimizationAlgorithm import baseIndividual, InterationResult, baseSIOA

class Salp(baseIndividual):
    '''
        樽海鞘个体类 \n
        :param search_space: 搜索空间 \n
    '''

    def update_head(self, best_position : np.ndarray, c1 : float):
        '''
            更新链头樽海鞘的位置
        '''
        # c2 = random.random()                        # 同一随机数
        c2 = np.random.random_sample(self.x_dim)    # 每一维度不同随机数
        c3 = random()
        second_part = c1 * (self.across * c2 + self.l_bound)
        
        if c3 < 0.5:
            self.position = best_position - second_part
        else:
            self.position = best_position + second_part

    def update_other(self, prev):
        '''
            更新链头樽海鞘的位置
        '''
        self.position += prev.position
        self.position /= 2


class SalpSwarm(baseSIOA):
    '''
        樽海鞘群类，要使用该优化算法，调用 iteration 方法即可 \n
        :param objective_func:  目标函数，参数为 x 向量 \n
        :param particle_num:    樽海鞘个数 \n
        :param search_space:    x 在各维度的取值范围，shape = (2, x_dim) \n
        :param constraint_func: 约束条件函数 返回值为 bool 类型\n
    '''
    
    individual_class_build_up_func = Salp

    def __init__(self, objective_func, salp_num : int, search_space : np.ndarray, constraint_func = None):
        super(SalpSwarm, self).__init__(
            objective_func, salp_num, search_space, constraint_func = constraint_func
        )

    def __build_up_swarm__(self):
        ''' 构建樽海鞘群 '''
        super(SalpSwarm, self).__build_up_swarm__()
        self.salp_swarm = self.individual_swarm     # 给群体起名，方便迭代写函数
        self.salp_num = self.individuals_num

    def c1_formula(self, t : int, T : int) -> float:
        '''
            返回当代 c1 值 \n
            这里使用非线性减少，可以更好为其他减少方式 \n
            :param t: 当前迭代次数 \n
            :param T: 最大迭代次数 \n
            :return float:
        '''
        return 2 * exp(- (4 * t / T) ** 2)

    def iteration(self, iter_num : int, if_show_process = True) -> InterationResult:
        '''
            樽海鞘群算法的迭代函数 \n
            :param iter_num: 最大迭代次数 \n
            :param if_show_process: 控制是否显示迭代进度，默认为显示 \n
            :return IterationResult:
        '''
        best_fitness_value = []
        
        self.get_fitness()
        best_salp_index = np.argmin(self.fitness)
        best_fitness_value.append(self.fitness[best_salp_index])
        best_position = self.salp_swarm[best_salp_index].position

        interator = tqdm(range(1, iter_num + 1)) if if_show_process else range(1, iter_num + 1)   # 根据 if_show_process 选择迭代器
        for t in interator:
            c_1 = self.c1_formula(t, iter_num)
            
            # 链头产生新解
            self.salp_swarm[0].update_head(best_position, c_1)

            # 其他产生新解
            for i in range(1, self.salp_num):
                self.salp_swarm[i].update_other(self.salp_swarm[i - 1])
            
            if self.constraint_func:
                self.constraint()
            else:
                self.bound_check()
            
            self.get_fitness()
            best_salp_index = np.argmin(self.fitness)
            best_fitness_value.append(self.fitness[best_salp_index])

            best_position = self.salp_swarm[best_salp_index].position.copy()

        
        return InterationResult(
                {
                    'best_position' : best_position,
                    'best_fitness' : self.fitness[best_salp_index],
                    'best_fitness_value_history' : best_fitness_value
                }
            )
    



if __name__ == '__main__':
    def cost(x : np.ndarray) -> float:
        return (x[0]** 2 - 2) ** 2
    
    search_space = np.array([
        [0],
        [2]
    ])
    solver = SalpSwarm(cost, 30, search_space)
    solution_dict = solver.iteration(100)
    solution = solution_dict.best_position
    print('solution = ', solution[0])