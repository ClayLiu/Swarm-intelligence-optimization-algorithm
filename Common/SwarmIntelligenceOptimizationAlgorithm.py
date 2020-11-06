import numpy as np 
from Common.constants import *

class InterationResult():
    def __init__(self, Result_dict : dict):
        self.best_position = Result_dict['best_position']
        self.best_fitness = Result_dict['best_fitness']
        self.best_fitness_value_history = Result_dict['best_fitness_value_history']

class baseIndividual():
    def __init__(self, search_space : np.ndarray):
        
        self.x_dim = search_space.shape[MatrixShapeIndex.column]
        self.l_bound = search_space[Bounds.lower]                          # x 在各维的下界
        self.u_bound = search_space[Bounds.upper]                          # x 在各维的上界
        self.across = search_space[Bounds.upper] - search_space[Bounds.lower]   # x 在各维的取值跨度
        
        self.position = self.across * np.random.random_sample(self.x_dim) + self.l_bound    # 在搜索空间均匀分布(非等距分布)散布个体

    def refresh(self, search_space : np.ndarray):
        self.__init__(search_space)

    def bound_check(self):
        bigger = self.position > self.u_bound
        smaller = self.position < self.l_bound

        self.position[bigger] = self.u_bound[bigger]
        self.position[smaller] = self.l_bound[smaller]
        

class baseSIOA():
    '''
    :param objective_func:  目标函数，参数为 x 向量 \n
    :param individuals_num: 个体数量个数 \n
    :param search_space:    x 在各维度的取值范围，shape = (2, x_dim) \n
    :param constraint_func: 约束条件函数 返回值为 bool 类型 \n
    '''

    individual_class_build_up_func = baseIndividual # 个体类的构造函数

    def __init__(self, objective_func, individuals_num, search_space, constraint_func = None):        
        self.objective_func = objective_func
        self.individuals_num = individuals_num
        self.search_space = search_space
        self.constraint_func = constraint_func

        self.__build_up_swarm__()

    
    def __build_up_swarm__(self):
        individual_swarm = []
        for i in range(self.individuals_num):
            temp_individual = self.individual_class_build_up_func(self.search_space)
            individual_swarm.append(temp_individual)
        # self.individual_swarm = np.array(individual_swarm)
        self.individual_swarm = individual_swarm          # 使用 list 以方便编程时获取元素 的属性
        # individual_swarm.clear()
        
        if self.constraint_func:
            self.constraint()

        self.fitness = np.zeros(self.individuals_num) + float('inf')

    def refresh_swarm(self):
        del self.individual_swarm
        self.__build_up_swarm__()
        

    def constraint(self):
        for individual in self.individual_swarm:
            while not self.constraint_func(individual.position):
                individual.refresh(self.search_space)
                
    def bound_check(self):
        for individual in self.individual_swarm:
            individual.bound_check()

    def get_fitness(self):
        for i, individual in enumerate(self.individual_swarm):
            self.fitness[i] = self.objective_func(individual.position.copy())
    
    def __get_next_generation__(self):
        pass

    def iteration(self, iter_num : int) -> InterationResult:
        pass

def Selection(Swarm : baseSIOA, have_fitness = True):
    '''
        选择机制
    '''
    if not have_fitness:
        Swarm.get_fitness()
    
    sorted_index = np.argsort(Swarm.fitness)
    
    best_half_index = sorted_index < (Swarm.individuals_num >> 1)
    worst_half_index = sorted_index >= (Swarm.individuals_num >> 1)

    # 用好的一半替代差的一半的位置
    for good, bad in zip(best_half_index, worst_half_index):
        Swarm.individual_swarm[bad].position = Swarm.individual_swarm[good].position
