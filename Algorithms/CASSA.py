import numpy as np 
from tqdm import tqdm

from Algorithms.SSA import Salp, SalpSwarm
from Common.SwarmIntelligenceOptimizationAlgorithm import InterationResult

class Salp_chaotic(Salp):
    def __tent_ize__(self, x : np.ndarray, u = 2) -> np.ndarray:
        where = x >= 0.5
        x[where] = 1 - x[where]
        x *= u
        return x

    def __init__(self, search_space : np.ndarray, crazy_probability = 0.3):
        super(Salp_chaotic, self).__init__(search_space)

        random_array = np.random.random_sample(self.x_dim)      # 使用 tent 映射初始化
        tent_ed = self.__tent_ize__(random_array)               # 
        self.position = self.across * tent_ed + self.l_bound    # 樽海鞘位置

        self.crazy_probability = crazy_probability                  # 疯狂概率
        self.x_craziness = 0.0001 + np.zeros_like(self.position)    # 不知为何参数      

    def update_head(self, best_position : np.ndarray, c1 : float):
        '''
            疯狂自适应樽海鞘 链头樽海鞘更新函数
            比标准樽海鞘多加一项
        '''
        super(Salp_chaotic, self).update_head(best_position, c1)
        c4 = np.random.random_sample(self.x_dim)

        P_c4 = c4 <= self.crazy_probability
        sign_c4 = np.ones(self.x_dim)
        where_half = c4 >= 0.5

        sign_c4[where_half] = -1

        self.position[P_c4] += sign_c4[P_c4] * self.x_craziness[P_c4]
    
    def update_other(self, prev, weight : float):
        self.position += weight * prev.position
        self.position /= 2


class SalpSwarm_chaotic(SalpSwarm):
    individual_class_build_up_func = Salp_chaotic

    def __init__(self, objective_func, salp_num : int, search_space : np.ndarray, crazy_probability = 0.3, weight = (0.9, 0.4), constraint_func = None):
        self.crazy_probability = crazy_probability
        self.start_weight, self.end_weight = weight
        assert self.start_weight > self.end_weight, '惯性权重参数输入错误'

        super(SalpSwarm_chaotic, self).__init__(
            objective_func, salp_num, search_space, constraint_func
        )
        
    def __build_up_swarm__(self):
        individual_swarm = []
        for i in range(self.individuals_num):
            temp_individual = self.individual_class_build_up_func(self.search_space, self.crazy_probability)
            individual_swarm.append(temp_individual)
        # self.individual_swarm = np.array(individual_swarm)
        self.individual_swarm = individual_swarm          # 使用 list 以方便编程时获取元素 的属性
        # individual_swarm.clear()
        
        if self.constraint_func:
            self.constraint()

        self.fitness = np.zeros(self.individuals_num) + float('inf')
        
        self.salp_swarm = self.individual_swarm     # 给群体起名，方便迭代写函数
        self.salp_num = self.individuals_num


    def current_weight_formula(self, t : int, T : int) -> float:
        '''
            返回当代 惯性权重 值 \n
            这里使用线性减少，可以更换为其他减少方式 \n
            \omega = \omega_s (\omega_s - \omega_e)(T - t)/T \n
            :param t: 当前迭代次数 \n
            :param T: 最大迭代次数 \n
            :return float:
        '''
        return self.start_weight * (self.start_weight - self.end_weight) * (T - t) / T


    def iteration(self, iter_num : int, if_show_process = True) -> InterationResult:
        best_fitness_value = []
        
        self.get_fitness()
        best_salp_index = np.argmin(self.fitness)
        best_fitness_value.append(self.fitness[best_salp_index])
        best_position = self.salp_swarm[best_salp_index].position
        half_num = (self.salp_num) >> 1

        interator = tqdm(range(1, iter_num + 1)) if if_show_process else range(1, iter_num + 1)   # 根据 if_show_process 选择迭代器
        for t in interator:
            c_1 = self.c1_formula(t, iter_num)
            current_weight = self.current_weight_formula(t, iter_num)

            # 链头产生新解
            for i in range(half_num):
                self.salp_swarm[i].update_head(best_position, c_1)

            # 其他产生新解
            for i in range(half_num, self.salp_num):
                self.salp_swarm[i].update_other(self.salp_swarm[i - 1], current_weight)
            
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


