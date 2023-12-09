"""
@name:TestFunction.py
@author:ZHANG Yunyi
@date:2023/12/6
@description:用于测试优化算法的10个标准函数
"""
import math
import numpy as np


def sphere(x):
    return sum([i**2 for i in x])


def rosenbrock(x):
    return sum([100*(x[i+1]-x[i]**2)**2 + (1-x[i])**2 for i in range(len(x)-1)])


def rastrigin(x):
    return 10*len(x) + sum([(i**2 - 10 * np.cos(2 * np.pi * i)) for i in x])


def griewank(x):
    return 1 + sum([(i**2)/4000 for i in x]) - np.prod([np.cos(i/np.sqrt(idx+1)) for idx, i in enumerate(x)])


def ackley(x):
    return -20*np.exp(-0.2*np.sqrt(sum([i**2 for i in x])/len(x))) - np.exp(sum([np.cos(2*np.pi*i) for i in x])/len(x)) + 20 + np.exp(1)


def zakharov(x):
    return sum([i**2 for i in x]) + sum([0.5*i*j for i, j in enumerate(x)])**2 + sum([0.5*i*j for i, j in enumerate(x)])**4


def michalewicz(x, m=10):
    return -sum([np.sin(i)*np.sin((j+1)*i**2/np.pi)**(2*m) for j, i in enumerate(x)])


def schwefel(x):
    return 418.9829*len(x) - sum([i*np.sin(np.sqrt(abs(i))) for i in x])


def dixon_price(x):
    return (x[0] - 1)**2 + sum([(i+1)*(2*x[i]**2 - x[i-1])**2 for i in range(1, len(x))])


def styblinski_tang(x):
    return sum([i**4 - 16*i**2 + 5*i for i in x])/2


test_function_list = [sphere, rosenbrock, rastrigin, griewank, ackley, zakharov, michalewicz, schwefel,
                      dixon_price, styblinski_tang]

test_function_name_list = ['sphere', 'rosenbrock', 'rastrigin', 'griewank', 'ackley', 'zakharov', 'michalewicz',
                           'schwefel', 'dixon_price', 'styblinski_tang']


