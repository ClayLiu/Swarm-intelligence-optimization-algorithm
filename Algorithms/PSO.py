import numpy as np 
from tqdm import tqdm

from Common.constants import *
from Common.SwarmIntelligenceOptimizationAlgorithm import baseIndividual, IterationResult, baseSIOA


class Particle(baseIndividual):
    '''
        粒子个体类 \n
        这里默认初始时粒子的速度为 0 \n
        c1, c2 为两个加速系数，即玄学超参数 \n
        :param search_space: 搜索空间 \n
    '''
    def __init__(self, search_space : np.ndarray):

        self.c1 = 2 # 加速系数
        self.c2 = 2 # 加速系数

        super(Particle, self).__init__(search_space)
        self.velocity = np.zeros_like(self.position)
        
        self.prev_best_position = self.position.copy()
        self.prev_best_fitness = 99999999999999

    def update_position_and_velocity(self, omega : float, global_best_position : np.ndarray):
        '''
            更新粒子的速度与位置 \n
            :param omega: 惯性权重 \n
            :param global_best_position: 群体最优的位置 \n
        '''
        self.velocity = omega * self.velocity + \
            self.c1 * np.random.random_sample(self.x_dim) * (self.prev_best_position - self.position) + \
            self.c2 * np.random.random_sample(self.x_dim) * (global_best_position - self.position)
        
        self.position += self.velocity
    
    def update_prev_best(self, fitness : float):
        '''
            如果当前的位置比历史最优位置好，则更新历史最优位置 \n
            :param fitness: 当前位置的适应度值 \n
        '''
        if self.prev_best_fitness > fitness:
            self.prev_best_fitness = fitness
            self.prev_best_position = self.position.copy()
    
    def refresh(self, search_space : np.ndarray):
        '''
            在搜索空间生成新的粒子
        '''
        self.__init__(search_space)


class ParticleSwarm(baseSIOA):
    '''
        粒子群类，要使用该优化算法，调用 iteration 方法即可 \n
        :param objective_func:  目标函数，参数为 x 向量 \n
        :param particle_num:    粒子个数 \n
        :param search_space:    x 在各维度的取值范围，shape = (2, x_dim) \n
        :param constraint_func: 约束条件函数 返回值为 bool 类型\n
    '''

    individual_class_build_up_func = Particle
    def __init__(self, objective_func, particle_num : int, search_space : np.ndarray, constraint_func = None):
        super(ParticleSwarm, self).__init__(
            objective_func, particle_num, search_space, constraint_func = constraint_func
        )
    

    def __build_up_swarm__(self):
        '''
            构建粒子群
        '''
        super(ParticleSwarm, self).__build_up_swarm__()
        self.particle_swarm = self.individual_swarm     # 改个名，方便写下面的程序
        self.particle_num = self.individuals_num
    
    def omega_formula(self, t : int, T : int) -> float:
        '''
            返回当代的惯性权重 omega \n
            这里使用线性减少，可以更换为其他的减小函数 \n
            :param t: 当前迭代次数 \n
            :param T: 最大迭代次数 \n
            :return float:
        '''
        return 0.9 - (t / (T - 1)) * 0.5    # 因为在 iteration 方法里的迭代次数 t 由 range() 生成，
                                            # 为序列 [0, ..., max_iter_num - 1]，所以这里 / (T - 1)

    def __get_next_generation__(self, omega : float):
        '''
            生成下一代群体
        '''
        
        for i, particle in enumerate(self.particle_swarm):
            particle.update_position_and_velocity(omega, self.global_best_position)
            particle.bound_check()

            # 若有约束条件，则判断更新后是否在可行域内，若不在则重置粒子
            if self.constraint_func:
                while not self.constraint_func(particle.position):
                    particle.refresh(self.search_space)

            fitness = self.objective_func(particle.position.copy())
            self.fitness[i] = fitness
            particle.update_prev_best(fitness)

            # 更新群体最优位置
            if self.global_best_fitness > fitness:
                self.global_best_fitness = fitness
                self.global_best_position = particle.position.copy()


    def iteration(self, iter_num : int, if_show_process = True) -> IterationResult:
        '''
            粒子群算法的迭代函数 \n
            :param iter_num: 最大迭代次数 \n
            :param if_show_process: 控制是否显示迭代进度，默认为显示 \n
            :return IterationResult:
        '''
        best_fitness_value = []

        self.get_fitness()                                              # 计算初始的各粒子的适应度
        best_particle_index = np.argmin(self.fitness)                   # 找到最优的粒子的索引
        best_fitness_value.append(self.fitness[best_particle_index])    # 记录最优适应度值    
        
        self.global_best_position = self.particle_swarm[best_particle_index].position.copy()
        self.global_best_fitness = self.fitness[best_particle_index]


        for particle, fitness in zip(self.particle_swarm, self.fitness):
            particle.update_prev_best(fitness)

        iterator = tqdm(range(iter_num)) if if_show_process else range(iter_num)   # 根据 if_show_process 选择迭代器
        for t in iterator:
            omega = self.omega_formula(t, iter_num)
            self.__get_next_generation__(omega)
            best_fitness_value.append(self.global_best_fitness)

        return IterationResult(
            {
                'best_position' : self.global_best_position,
                'best_fitness' : self.global_best_fitness,
                'best_fitness_value_history' : best_fitness_value
            }
        )            
