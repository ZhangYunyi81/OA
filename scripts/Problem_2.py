"""
@name:Problem_2.py
@author:ZHANG Yunyi
@date:2023/12/6
@description:采用粒子群算法求解实际优化问题1(模型参数辨识)
"""
import matplotlib.pyplot as plt

from PSOAlgorithm import *
import numpy as np
import matplotlib.pyplot as plt
# 步长
STEP = 0.1
# 模型输入输出观测数据
OBSERVE_DATA = np.loadtxt('../data/observe_data.txt')
# 数据个数
N = OBSERVE_DATA.shape[1]
# PSO空间维数
DIMENSION = 4
# 粒子个数
POPULATION = 1000
# PSO迭代次数
ITERATION = 100
def read_observe_data(file_path):
    data = np.loadtxt(file_path, dtype=np.float32)
    data = np.array(data)
    return data[0],data[1]

def sgn(x):
    if x > 0:
        return 1
    if x < 0:
        return -1
    if x == 0:
        return 0

def cal_obj_func(particle_position):
    # gp = particle_position[0]
    # hp = particle_position[1]
    # k1p = particle_position[2]
    # k2p = particle_position[3]
    # xmin = -4
    # xmax = 4
    # x = []
    # y_estimate = []
    y_estimate = estimate_result(particle_position)
    y_observe = OBSERVE_DATA[1]
    # for i in range(N):
    #     x.append(xmin + i * STEP)
    #     x_abs = abs(x[i])
    #     if x_abs <= gp:
    #         y_estimate.append(0)
    #     elif gp < x_abs <= hp:
    #         y_estimate.append(k1p * (x[i] - gp * sgn(x[i])))
    #     elif x_abs > hp:
    #         y_estimate.append(k2p*(x[i]-hp*sgn(x[i]))+k1p*(hp-gp)*sgn(x[i]))
    y_diff = np.array(y_estimate) - y_observe
    J = 0.5 * np.linalg.norm(y_diff)
    return J


def estimate_result(particle_position):
    gp = particle_position[0]
    hp = particle_position[1]
    k1p = particle_position[2]
    k2p = particle_position[3]
    xmin = -4
    xmax = 4
    x = []
    y_estimate = []
    y_observe = OBSERVE_DATA[1]
    for i in range(N):
        x.append(xmin + i * STEP)
        x_abs = abs(x[i])
        if x_abs <= gp:
            y_estimate.append(0)
        elif gp < x_abs <= hp:
            y_estimate.append(k1p * (x[i] - gp * sgn(x[i])))
        elif x_abs > hp:
            y_estimate.append(k2p * (x[i] - hp * sgn(x[i])) + k1p * (hp - gp) * sgn(x[i]))
    return y_estimate

if __name__ == '__main__':
    fig_iter = plt.figure()
    pso_optimizer = PSO(dim=DIMENSION, n_particles=POPULATION, bounds=(0, 5), iteration=ITERATION, func_list=[cal_obj_func])
    print('正在进行参数辨识，迭代进度为：')
    best_position, best_score = pso_optimizer.optimize(func_idx=0)
    print('Best solution is :')
    print('g = ',best_position[0])
    print('h = ', best_position[1])
    print('k1 = ', best_position[2])
    print('k2 = ', best_position[3])
    pso_optimizer.draw_value_curve('')
    plt.title('Model parameters identification')
    plt.xlabel('number of iteration')
    plt.ylabel('minimum squared error')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig_observe = plt.figure()
    plt.subplot(1,2,1)
    plt.plot(OBSERVE_DATA[0], OBSERVE_DATA[1])
    plt.xlabel('input')
    plt.ylabel('output')
    plt.title('Observed')
    plt.subplot(1,2,2)
    plt.plot(OBSERVE_DATA[0], estimate_result(best_position), color='red')
    plt.xlabel('input')
    plt.ylabel('output')
    plt.title('Estimated')
    plt.show()


