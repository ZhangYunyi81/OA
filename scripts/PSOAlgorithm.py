import numpy as np
import matplotlib.pyplot as plt
from tqdm import *

class Particle:
    def __init__(self, dim, minx, maxx):
        self.position = np.random.uniform(low=minx, high=maxx, size=dim)
        self.velocity = np.random.uniform(low=-0.1, high=0.1, size=dim)
        self.best_position = np.copy(self.position)
        self.best_value = np.inf


class PSO:
    def __init__(self, dim, n_particles, bounds, iteration, func_list):
        self.dim = dim
        self.n_particles = n_particles
        self.bounds = bounds
        self.iteration = iteration
        self.particles = [Particle(dim, *bounds) for _ in range(n_particles)]
        self.global_best_value = np.inf
        self.best_value_list = []
        self.global_best_position = np.zeros(dim)
        self.objective_function_list = func_list

    def update_velocity(self, particle):
        w = 0.7  # 惯性权重
        c1 = 1.3  # 个体学习因子
        c2 = 1.7  # 社会学习因子
        particle.velocity = (w*particle.velocity +
                             c1*np.random.rand(self.dim)*(particle.best_position - particle.position) +
                             c2*np.random.rand(self.dim)*(self.global_best_position - particle.position))

    def update_position(self, particle):
        particle.position += particle.velocity
        particle.position = np.clip(particle.position, *self.bounds)

    def optimize(self, func_idx):
        self.best_value_list = []
        for i in tqdm(range(self.iteration)):
            for particle in self.particles:
                value = self.objective_function_list[func_idx](particle.position)
                if value < particle.best_value:
                    particle.best_score = value
                    particle.best_position = np.copy(particle.position)
                if value < self.global_best_value:
                    self.global_best_value = value
                    self.global_best_position = np.copy(particle.position)
            for particle in self.particles:
                self.update_velocity(particle)
                self.update_position(particle)
            self.best_value_list.append(self.global_best_value)
        return self.global_best_position, self.global_best_value

    def draw_value_curve(self, func_name):
        # fig = plt.figure()
        x = [i for i in range(self.iteration)]
        y = self.best_value_list
        plt.plot(x, y)
        plt.title(func_name, fontdict={'fontsize':7})
        plt.xticks(fontsize=3)
        plt.yticks(fontsize=3)
        # plt.show()


# 使用PSO
# dim = 2
# n_particles = 100
# bounds = (-10, 10)
# iterations = 100
#
# pso = PSO(dim, n_particles, bounds)
# best_position, best_score = pso.optimize(iterations)
#
# print("最优位置：", best_position)
# print("最优得分：", best_score)
