'''
    提供一些目标函数
'''

import numpy as np 
from math import sin, cos, exp, sqrt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Common.constants import *


class base_unconstrained_objective_func():
    '''
        无约束目标函数基类
    '''
    search_space_tuple = (-1, 1)
    func_formula_str = 'f(x)'

    def __init__(self, n : int = 2):
        self.x_dim = n
        self.search_space = np.vstack((
            np.ones(n, dtype = float),
            np.ones(n, dtype = float)
        ))

        self.search_space[Bounds.lower] = self.search_space_tuple[Bounds.lower]
        self.search_space[Bounds.upper] = self.search_space_tuple[Bounds.upper]

        self.best_f_x = 0
        self.best_x = None

    def func(self, x : np.ndarray) -> float:
        '''
            目标函数
        '''
        assert len(x) == self.x_dim, '向量维度不匹配！'
        assert x.dtype == np.float, '输入数据应是 float 类型'
        return 0


    def plot(self, step_length : int = 128, save_figure_path : str = None, dpi : int = 300, area : tuple = None):
        '''
            画出函数曲面 \n
            :param step_length: 画图步长 \n
            :param save_figure_path: 图像保存路径 \n
            :param dpi: 图像 dpi \n
            :param area: 画图区域 如 (-10, 10) 则绘画 x \in (-10, 10) , y \in (-10, 10) 的区域
        '''
        x_dim_save = self.x_dim
        self.__init__(2)

        if area:
            assert isinstance(area, tuple), '绘图空间必须为元组！'
            search_space_save = self.search_space.copy()
            self.search_space_tuple = area
            
            self.search_space[Bounds.lower] = self.search_space_tuple[Bounds.lower]
            self.search_space[Bounds.upper] = self.search_space_tuple[Bounds.upper]

        u, v = np.meshgrid(
            np.linspace(self.search_space_tuple[Bounds.lower], self.search_space_tuple[Bounds.upper], step_length),
            np.linspace(self.search_space_tuple[Bounds.lower], self.search_space_tuple[Bounds.upper], step_length)
        )

        z = np.zeros_like(u)

        for i in range(step_length):
            for j in range(step_length):
                x, y = u[i][j], v[i][j]
                z[i][j] = self.func(
                    np.array([x, y], dtype = float)
                )
        
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_surface(u, v, z, cmap = plt.cm.RdYlGn)
        plt.title('${}$'.format(self.func_formula_str))
        
        if save_figure_path:
            plt.savefig(save_figure_path, dpi = dpi)

        plt.show()

        # 还原现场
        self.__init__(x_dim_save)
        if area:
            self.search_space = search_space_save


''' 单峰函数 '''

class Sphere(base_unconstrained_objective_func):
    r'''
        目标函数为
        f(x) = \sum_{i = 1}^n x_i^2
    '''
    search_space_tuple = (-100, 100)
    func_formula_str = r'f(x) = \sum_{i = 1}^n x_i^2'


    def func(self, x : np.ndarray) -> float:
        r'''
            目标函数为
            f(x) = \sum_{i = 1}^n x_i^2
        '''
        
        super(Sphere, self).func(x)
        return np.sum(x ** 2)


class Step(base_unconstrained_objective_func):
    r'''
        目标函数为
        f(x) = \sum_{i = 1}^n\lfloor x_i + 0.5 \rfloor ^2
    '''

    search_space_tuple = (-100, 100)
    func_formula_str = r'f(x) = \sum_{i = 1}^n\lfloor x_i + 0.5 \rfloor ^2'

    def func(self, x : np.ndarray) -> float:
        r'''
            目标函数为
            f(x) = \sum_{i = 1}^n\lfloor x_i + 0.5 \rfloor ^2
        '''
        super(Step, self).func(x)
        floor = np.floor(x + 0.5)

        return np.sum(floor ** 2)


class Schwefel_1_2(base_unconstrained_objective_func):
    search_space_tuple = (-100, 100)
    func_formula_str = r'f(x) = \sum_{i = 1}^n\left(\sum_{j=1}^i x_j\right)^2'
    
    def func(self, x : np.ndarray) -> float:
        r'''
            目标函数为
            f(x) = \sum_{i = 1}^n\left(\sum_{j=1}^i x_j\right)^2
        '''
        super(Schwefel_1_2, self).func(x)
        x_copy = x.copy()
        for i in range(1, len(x)):
            x_copy[i] += x_copy[i - 1]
        
        return np.sum(x_copy ** 2)


class Schwefel_2_21(base_unconstrained_objective_func):
    search_space_tuple = (-100, 100)
    func_formula_str = r'f(x) = \max_{i=1}^n\left\{|x_i|\right\}'
    
    def func(self, x : np.ndarray) -> float:
        r'''
            目标函数为
            f(x) = \max_{i=1}^n\left\{|x_i|\right\}
        '''
        super(Schwefel_2_21, self).func(x)
        return np.max(np.abs(x))


class Schwefel_2_22(base_unconstrained_objective_func):
    search_space_tuple = (-10, 10)
    func_formula_str = r'f(x) = \sum_{i = 0}^n |x_i| + \prod_{i = 0}^n |x_i|'

    def func(self, x : np.ndarray) -> float:
        r'''
            目标函数为
            f(x) = \sum_{i = 0}^n |x_i| + \prod_{i = 0}^n |x_i|
        '''
        super(Schwefel_2_22, self).func(x)
        abs_x = np.abs(x)
        return np.sum(abs_x) + np.prod(abs_x)


class Rosenbrock(base_unconstrained_objective_func):
    search_space_tuple = (-2.048, 2.048)
    func_formula_str = r'f(x) = \sum_{i = 1}^{\frac{n}{2}}\left(100\left(x_{2i} - x_{2i - 1}^2\right)^2 + \left(1 - x_{2i - 1}\right)^2\right)'

    def __init__(self, n : int = 2):
        assert not n & 1, '该目标函数变量个数必须为偶数'

        super(Rosenbrock, self).__init__(n)

        self.index = np.arange(0, self.x_dim, 2)
        self.index_plus_1 = self.index + 1
        
        self.best_x = np.ones(self.x_dim, dtype = float)
        self.best_f_x = 0

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


''' 多峰函数 '''

class Rastrigin(base_unconstrained_objective_func):
    search_space_tuple = (-5.12, 5.12)
    func_formula_str = r'f(x) = \sum_{i=1}^{n}\left(x_{i}^{2}-10 \cos \left(2 \pi x_{i}\right)+10\right)'

    def func(self, x : np.ndarray) -> float:
        r'''
            目标函数为
            f(x) = \sum_{i=1}^{n}\left(x_{i}^{2}-10 \cos \left(2 \pi x_{i}\right)+10\right)
        '''
        super(Rastrigin, self).func(x)
        return np.sum(x ** 2 - 10 * np.cos(2 * pi * x) + 10)
    

class Schwefel_2_26(base_unconstrained_objective_func):
    search_space_tuple = (-500, 500)
    func_formula_str = r'f(x) = 418.9829 n+\sum_{i=1}^{n} x_{i} \sin \left(\sqrt{\left|x_{i}\right|}\right)'

    def __init__(self, n : int = 2):
        super(Schwefel_2_26, self).__init__(n)
        self.constant = 418.9829 * n
        
        self.best_x = np.zeros(self.x_dim) + -420.9687

    def func(self, x : np.ndarray) -> float:
        r'''
            目标函数为
            f(x) = 418.9829 n+\sum_{i=1}^{n} x_{i} \sin \left(\sqrt{\left|x_{i}\right|}\right)
        '''
        super(Schwefel_2_26, self).func(x)
        return self.constant + np.sum(x * np.sin(np.sqrt(np.abs(x))))


class Griewank(base_unconstrained_objective_func):
    search_space_tuple = (-600, 600)
    func_formula_str = r'f(x) = \frac{1}{4000} \sum_{i=1}^{n} x_{i}^{2}-\prod_{i=1}^{n} \cos \left(\frac{x_{i}}{\sqrt{i}}\right)+1'

    def __init__(self, n : int = 2):
        super(Griewank, self).__init__(n)
        i = np.arange(1, self.x_dim + 1)
        self.sqrt_i = np.sqrt(i)

    def func(self, x : np.ndarray) -> float:
        r'''
            目标函数为
            f(x) = \frac{1}{4000} \sum_{i=1}^{n} x_{i}^{2}-\prod_{i=1}^{n} \cos \left(\frac{x_{i}}{\sqrt{i}}\right)+1
        '''
        super(Griewank, self).func(x)
        return (np.sum(x ** 2) / 4000) - np.prod(np.cos(x / self.sqrt_i)) + 1

    
class Ackley(base_unconstrained_objective_func):
    r'''
        目标函数为
        f(x) = -20\exp\left(-0.2\sqrt{\sum_{i=1}^n\frac{x_i^2}{n}}\right) - \exp\left(\sum_{i = 1}^n\frac{\cos 2\pi x_i}{n}\right) + 20 + e
    '''

    search_space_tuple = (-32, 32)
    func_formula_str = r'f(x) = -20\exp\left(-0.2\sqrt{\sum_{i=1}^n\frac{x_i^2}{n}}\right) - \exp\left(\sum_{i = 1}^n\frac{\cos 2\pi x_i}{n}\right) + 20 + e'

    def func(self, x : np.ndarray) -> float:
        r'''
            目标函数为
            f(x) = -20\exp\left(-0.2\sqrt{\sum_{i=1}^n\frac{x_i^2}{n}}\right) - \exp\left(\sum_{i = 1}^n\frac{\cos 2\pi x_i}{n}\right) + 20 + e
        '''
        super(Ackley, self).func(x)
        return -20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / self.x_dim)) - np.exp(np.sum(np.cos(2 * pi * x)) / self.x_dim) + 20 + e
        

class Foxholes(base_unconstrained_objective_func):
    search_space_tuple = (-65.56, 65.56)
    func_formula_str = r'f(x) = \left[\frac{1}{500} + \sum_{j = 1}^{25}\frac{1}{j + \sum_{i = 1}^2(x_i-a_{ij})^6}\right]^{-1}'
    def __init__(self, n : int = 2):
        super(Foxholes, self).__init__(2)
        self.a = np.array([
            [-32, -16, 0, 16, 32] * 5,
            [-32] * 5 + [-16] * 5 + [0] * 5 + [16] * 5 + [32] * 5
        ])
        
        self.best_x = np.zeros(2) - 32
        self.best_f_x = self.func(self.best_x)

    def func(self, x : np.ndarray) -> float:
        super(Foxholes, self).func(x)

        j = np.arange(1, 26)
        x_a = x - self.a.T
        sum_x_a_pow_6 = np.sum(x_a ** 6, axis = 1)
        sum_part_out = np.sum(1 / (j + sum_x_a_pow_6))

        return 1 / ((1 / 500) + sum_part_out)
        

class Schaffer(base_unconstrained_objective_func):
    search_space_tuple = (-100, 100)
    def __init__(self, n : int = 2):
        super(Schaffer, self).__init__(2)        

        self.best_f_x = -1
        self.best_x = np.zeros(2)

    def func(self, x : np.ndarray) -> float:
        super(Schaffer, self).func(x)
        sum_square_x = np.sum(x ** 2)
        return ((sin(sqrt(sum_square_x)) ** 2 - 0.5) / ((1 + 0.001 * sum_square_x) ** 2)) - 0.5
