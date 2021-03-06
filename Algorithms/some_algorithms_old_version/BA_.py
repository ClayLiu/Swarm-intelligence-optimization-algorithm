import numpy as np
import random
import math
import tqdm
from Common.constants import *

from Common.SwarmIntelligenceOptimizationAlgorithm import InterationResult

class Bat():
    def __init__(self, position : np.ndarray, velocity : np.ndarray, f_bound : tuple):
        
        self.position = position
        self.velocity = velocity
        # self.A = random.random()     # 响度   [0, 1]
        self.A = 1.0    # A == 1 also can be used
        self.r = random.random()     # 脉冲率 [0, 1]       
        self.r_zero = self.r

        self.min_f = f_bound[Bounds.lower]
        self.max_f = f_bound[Bounds.upper]
        self.max_min_f = self.max_f - self.min_f

        self.frequency = random.random() * self.max_min_f + self.min_f
        self.position_new = self.position

    def update_global(self, best_position : np.ndarray):
        beta = random.random()
        self.frequency = self.min_f + self.max_min_f * beta

        self.velocity += (self.position - best_position) * self.frequency
        self.position_new = self.velocity + self.position
    
    def search_local(self, ave_A : float) -> np.ndarray:
        eps = (random.random() * 2 - 1)
        self.position_new = self.position + eps * ave_A

    def update_A_and_r(self, alpha : float, gamma : float, t : int):
        self.A *= alpha
        self.r = self.r_zero * (1 - math.exp(- gamma * t))

    def copy(self):
        return Bat(
            self.position.copy(),
            self.velocity.copy(),
            self.f_bound
        )

class BatSwarm():
    def __init__(self, func, constraintFunction, batQuantity, alpha, gamma, x_bounds_list, v_bounds_list, f_bound : tuple):
        """
        :param func:                目标函数 \n
        :param constrainFunction:   约束条件判断函数 若为 None 则表示无约束\n
        :param batQuantity:         蝙蝠个数 \n
        :param alpha:               alpha常数 (0, 1) \n
        :param gamma:               gamma常数 (0, +∞) \n
        :param x_bounds_list:       x在各维度的取值范围 \n
        :param v_bounds_list:       v在各维度的取值范围 \n
        :param f_bound:             f的取值范围 \n
        """

        self.func = func
        self.constraintFunction = constraintFunction
        self.batQuantity = batQuantity
        self.x_bounds_list = x_bounds_list
        self.v_bounds_list = v_bounds_list
        self.f_bound = f_bound
        self.x_dim = len(x_bound)

        self.alpha = alpha
        self.gamma = gamma

        self.batFrequecy = (np.random.random(batQuantity) * (f_bound[Bounds().upper] - f_bound[Bounds().lower])) + f_bound[Bounds().lower]
        self.batPositions = generate_population(batQuantity, x_bounds_list, constraintFunction)
        self.batVelocitys = self.generate_velocitys(batQuantity, v_bounds_list)
        self.batPulseRate = np.random.random(batQuantity)
        self.batPulseRate_zero = self.batPulseRate.copy()
        self.batLoudness = np.ones(batQuantity, dtype=np.float)

        self.fitness = np.zeros(self.bat_num) + 1e10
    
    def generate_velocitys(self, batQuantity, boundsList):
        span = []
        lower = []
        for bound in boundsList:
            span.append(bound[Bounds().upper] - bound[Bounds().lower])
            lower.append(bound[Bounds().lower])
        
        span = np.array(span)
        lower = np.array(lower)

        velocitys = []
        for i in range(batQuantity):
            singleBatVelocity = np.random.random(self.x_dim) * span + lower
            velocitys.append(singleBatVelocity)
        
        return np.vstack(velocitys)

    def __in_bound_check(self, x : np.ndarray):
        for i in range(self.x_dim):
            if x[i] > self.x_bound[i][Bounds.upper]:
                return False
            elif x[i] < self.x_bound[i][Bounds.lower]:
                return False

    def get_fitness(self):
        for batPosition in self.batPositions:
            self.fitness[i] = self.func(bat.position)

    def iteration(self, iter_num : int):
        
        best_fitness_value = []
        
        self.get_fitness()
        best_bat_index = np.argmin(self.fitness)
        best_fitness_value.append(self.fitness[best_bat_index])
        best_position = self.bat_swarm[best_bat_index].position.copy()
        print(best_position, self.fitness[best_bat_index])

        for t in tqdm.tqdm(range(1, iter_num + 1)):
            _sum = 0.0
            for single_bat in self.bat_swarm:
                _sum += single_bat.A
            current_ave_A = _sum / self.bat_num

            for i, single_bat in enumerate(self.bat_swarm):
                subject_pass = False
                # 产生新解
                single_bat.update_global(best_position)

                if random.random() > single_bat.r:
                    single_bat.search_local(current_ave_A)

                if random.random() < single_bat.A:
                    # 判断是否符合约束条件
                    if self.subject_func:
                        subject_pass = self.subject_func(single_bat.position_new) and self.__in_bound_check(single_bat.position_new)
                    else:
                        subject_pass = self.__in_bound_check(single_bat.position_new)
                    
                    # 新解满足约束条件
                    if subject_pass:
                        new_fitness = self.func(single_bat.position_new)
                        if new_fitness < self.fitness[i]:
                            self.fitness[i] = new_fitness
                            single_bat.position = single_bat.position_new.copy()
                            
                            if new_fitness < self.fitness[best_bat_index]:
                                best_bat_index = i
                                best_position = self.bat_swarm[best_bat_index].position  # 改变当代最优蝙蝠，代替迭代后再重新选出最优蝙蝠的低效做法
                            
                            single_bat.update_A_and_r(self.alpha, self.gamma, t)    
            
            # self.subjections()    # 因为始解满足约束条件，每次新解也检查一遍，所以不需要再全部检查了，减少运算量
            # self.get_fitness()    # 因为产生新解时已经算了一遍适应度，所以不需要再全部计算了，避免重复

            best_bat_index = np.argmin(self.fitness)
            best_fitness_value.append(self.fitness[best_bat_index])
            # best_position = self.bat_swarm[best_bat_index].position.copy()
        

        return InterationResult(
                {
                    'best_position' : best_position,
                    'best_fitness' : self.fitness[best_bat_index],
                    'best_fitness_value_history' : best_fitness_value
                }
            )