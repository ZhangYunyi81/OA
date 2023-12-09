"""
@name:Problem1.py
@author:ZHANG Yunyi
@date:2023/12/6
@description: 粒子群算法求解10种标准测试函数
"""
import random
import math
import time
import matplotlib.pyplot as plt
import pyswarms as ps
import numpy as np
from PSOAlgorithm import *
from TestFunction import *
# 定义10个测试函数
if __name__  == '__main__':
    # 初始化粒子群优化器
    fig = plt.figure()
    for i in range(len(test_function_list)):
        print('------------------------------------------------')
        print('             正在求解第'+str(i+1)+'个函数')
        print('------------------------------------------------')
        pso_optimizer = PSO(dim=4, n_particles=100, bounds=(-10, 10), iteration=100, func_list=test_function_list)
        begin_time = time.time()
        best_position, best_score = pso_optimizer.optimize(func_idx=i)
        end_time = time.time()
        plt.subplot(2,5,i+1)
        pso_optimizer.draw_value_curve(test_function_name_list[i])
        cal_time = end_time - begin_time
        print(test_function_name_list[i]+'函数最优值为:', round(best_score,4))
        print(test_function_name_list[i]+'函数最优位置为:', [round(p, 4) for p in best_position])
        print(test_function_name_list[i]+'函数计算时长为:', round(cal_time, 4))
    plt.show()