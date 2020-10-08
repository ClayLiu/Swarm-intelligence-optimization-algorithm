'''
    提供一些目标函数
'''

import numpy as np 
from math import pi
from Common.constants import *

class base_unconstrained_objective_func():
    '''
        无约束目标函数基类
    '''
    def __init__(self, n : int):
        self.x_dim = n
        self.search_space = np.vstack((
            np.ones(n, dtype = float),
            np.ones(n, dtype = float)
        ))
    
    def func(self, x : np.ndarray) -> float:
        '''
            目标函数
        '''
        assert len(x) == self.x_dim, '向量维度不匹配！'


class Sphere(base_unconstrained_objective_func):
    def __init__(self, n : int):
        super(Sphere, self).__init__(n)
        self.search_space[Bounds.lower] *= -100
        self.search_space[Bounds.upper] *= 100 


    def func(self, x : np.ndarray) -> float:
        r'''
            目标函数为
            f(x) = \sum_{i = 1}^n x_i^2
        '''
        super(Sphere, self).func(x)
        return np.sum(x ** 2)


class Rosenbrock(base_unconstrained_objective_func):
    def __init__(self, n : int):
        assert not n & 1, '该目标函数变量个数必须为偶数'

        super(Rosenbrock, self).__init__(n)
        self.search_space[Bounds.lower] *= -2.048
        self.search_space[Bounds.upper] *= 2.048

        self.index = np.arange(0, self.x_dim, 2)
        self.index_plus_1 = self.index + 1

    def func(self, x : np.ndarray) -> float:
        r'''
            目标函数为
            f(x) = \sum_{i = 1}^{\frac{n}{2}}\left(100\left(x_{2i} - x_{2i - 1}^2\right)^2
                    + \left(1 - x_{2i - 1}\right)^2\right)
        '''
        super(Rosenbrock, self).func(x)
        
        vector_2i_1 = x[self.index]
        vector_2i = x[self.index_plus_1]

        return np.sum(100 * (vector_2i - vector_2i_1 ** 2) ** 2 + (1 - vector_2i_1) ** 2)

    
class Griewank(base_unconstrained_objective_func):
    def __init__(self, n : int):
        super(Griewank, self).__init__(n)
        self.search_space[Bounds.lower] *= -600
        self.search_space[Bounds.upper] *= 600
        i = np.arange(1, self.x_dim + 1)
        self.sqrt_i = np.sqrt(i)

    def func(self, x : np.ndarray) -> float:
        r'''
            目标函数为
            f(x) = \frac{1}{4000} \sum_{i=1}^{n} x_{i}^{2}-\prod_{i=1}^{n} \cos \left(\frac{x_{i}}{\sqrt{i}}\right)+1
        '''
        super(Griewank, self).func(x)
        return (np.sum(x ** 2) - np.prod(np.cos(x / self.sqrt_i))) / 4000


class Rastrigin(base_unconstrained_objective_func):
    def __init__(self, n : int):
        super(Rastrigin, self).__init__(n)
        self.search_space[Bounds.lower] *= -5.12
        self.search_space[Bounds.upper] *= 5.12

    def func(self, x : np.ndarray) -> float:
        r'''
            目标函数为
            f(x) = \sum_{i=1}^{n}\left(x_{i}^{2}-10 \cos \left(2 \pi x_{i}\right)+10\right)
        '''
        super(Rastrigin, self).func(x)
        return np.sum(x ** 2 - 10 * np.cos(2 * pi * x) + 10)
    

class Schwefel(base_unconstrained_objective_func):
    def __init__(self, n : int):
        super(Schwefel, self).__init__(n)
        self.search_space[Bounds.lower] *= -500
        self.search_space[Bounds.upper] *= 500
        self.constant = 418.9829 * n

    def func(self, x : np.ndarray) -> float:
        r'''
            目标函数为
            f(x) = 418.9829 n+\sum_{i=1}^{n} x_{i} \sin \left(\sqrt{\left|x_{i}\right|}\right)
        '''
        super(Schwefel, self).func(x)
        return self.constant + np.sum(x * np.sin(np.sqrt(np.abs(x))))
