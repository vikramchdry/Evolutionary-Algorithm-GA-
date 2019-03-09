#Genetic Algoritms using path finding


import numpy as np 
import matplotlib.pyplot as plt 


N_MOVES = 150
DNA_SIZE = N_MOVES*2         # 40 x moves, 40 y moves
DIRECTION_BOUND = [0, 1]
CROSS_RATE = 0.8
MUTATE_RATE = 0.0001
POP_SIZE = 100
N_GENERATIONS = 100
GOAL_POINT = [10, 5]
START_POINT = [0, 5]
OBSTACLE_LINE = np.array([[5, 2], [5, 8]])



class GeneticAlgorithm():
    def __init__(self,dna_size,dna_bound,cross_rate, mutate_rate,population_size):
        self.dna_bound = dna_bound
        dna_bound[1] += 1
        self.dna_size = dna_size
        self.cross_rate = cross_rate
        self.mutate_rate = mutate_rate
        self.population_size = population_size
        self.pop = np.random.randint(*dna_bound, size=(population_size, dna_size))

    def DNA2product(self, DNA, n_moves, start_point):
        pop = (DNA - 0.5) / 2
        pop[:, 0], pop[:, n_moves] = start_point[0], start_point[1]
        loc_x = np.cumsum(pop[:, :n_moves], axis=1)
        loc_y = np.cumsum(pop[:, n_moves:], axis=1)
        return loc_x, loc_y



    def fitness_function(self,loc_x,loc_y,goal_point,obstacle):
        dist_to_goal = np.sqrt((goal_point[0]-loc_x[:,-1])**2 + (goal_point[1] - loc_y[:,-1])**2)
        fitness = np.power(1/(dist_to_goal + 1),2)
        points = (loc_x > obstacle[0, 0] - 0.5) & (loc_x < obstacle[1, 0] + 0.5)
        y_values = np.where(points, loc_y, np.zeros_like(loc_y) - 100)
        bad_lines = ((y_values > obstacle[0, 1]) & (y_values < obstacle[1, 1])).max(axis=1)
        fitness[bad_lines] = 1e-6
        return fitness

    def selection(self,fitness):
        idx = np.random.choice(np.arange(self.population_size),size = self.population_size,replace = True, p = fitness/fitness.sum())
        return self.pop[idx]


    def crossover(self,parent,pop):
        if np.random.rand()<self.cross_rate:
            i_ = np.random.randint(0, self.population_size,size = 1)
            cross_points = np.random.randint(0,2,self.dna_size).astype(np.bool)
            parent[cross_points] = pop[i_, cross_points]
        return parent


    def mutate(self, child):
        for point in range(self.dna_size):
            if np.random.rand() < self.mutate_rate:
                child[point] = np.random.randint(*self.dna_bound) 
        return child

    def evolve(self, fitness):
        pop = self.selection(fitness)
        pop_copy = pop.copy()
        for parent in pop:
            child = self.crossover(parent, pop_copy)
            child = self.mutate(child)
            parent[:] = child
        self.pop = pop



class Line:
    def __init__(self,n_moves,goal_point,start_point,obstacle):
        self.n_moves = n_moves
        self.goal_point = goal_point
        self.start_point = start_point
        self.obstacle  = obstacle

        plt.ion()

    def plotting(self,loc_x,loc_y):
        plt.cla()
        plt.scatter(*self.goal_point, s = 200, c= 'r')
        plt.scatter(*self.start_point, s = 200, c= 'b')
        plt.plot(self.obstacle[:, 0], self.obstacle[:, 1], lw=3, c='k')
        plt.plot(loc_x.T, loc_y.T, c='k')
        plt.xlim((-5, 15))
        plt.ylim((-5, 15))
        plt.pause(0.01)



gene = GeneticAlgorithm(dna_size=DNA_SIZE, dna_bound=DIRECTION_BOUND,
        cross_rate=CROSS_RATE, mutate_rate=MUTATE_RATE, population_size=POP_SIZE)
env = Line(N_MOVES, GOAL_POINT, START_POINT, OBSTACLE_LINE)

for generation in range(N_GENERATIONS):
    lx, ly = gene.DNA2product(gene.pop, N_MOVES, START_POINT)
    fitness = gene.fitness_function(lx, ly, GOAL_POINT, OBSTACLE_LINE)
    gene.evolve(fitness)
    print('Gen:', generation, '| best fit:', fitness.max())
    env.plotting(lx, ly)

plt.ioff()
plt.show()













a = GeneticAlgorithm(100,[0,1],0.005, 0.01,1000)