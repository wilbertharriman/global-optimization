import numpy as np
import random
import math
import copy
# you must use python 3.6, 3.7, 3.8, 3.9 for sourcedefender
import sourcedefender
from HomeworkFramework import Function

class RS_optimizer(Function): # need to inherit this class "Function"
    def __init__(self, target_func):
        super().__init__(target_func) # must have this init to work normally

        self.lower = self.f.lower(target_func)
        self.upper = self.f.upper(target_func)
        self.dim = self.f.dimension(target_func)

        self.target_func = target_func

        self.eval_times = 0
        self.optimal_value = float("inf")
        self.optimal_solution = np.empty(self.dim)
        

    def get_optimal(self):
        return self.optimal_solution, self.optimal_value

    def checkBoundary(self, tmp):
        if tmp < self.lower:
            return min(self.upper, 2*self.lower-tmp)
        if tmp > self.upper:
            return max(self.lower, 2*self.upper-tmp)
        return tmp

    def JADE(self, maxFES, NP=20):
        meanCR = 0.5
        meanF = 0.5
        dimension = self.dim

        # Randomly generate initial population
        P = np.random.uniform(low=self.lower, high=self.upper, size=(NP, dimension))

        # Evaluate objective function
        results = []

        for i in range(NP):
            results.append(float(self.f.evaluate(func_num, P[i])))

        self.eval_times = NP
        while self.eval_times < maxFES:
            Scr = []
            Sf = []
            for i in range(NP):
                # print('=====================FE=====================')
                # print(self.eval_times)
                CR = np.random.normal(meanCR, 0.1)
                F = np.random.normal(meanF, 0.1)
                # Generate three offsprings with mutation
                candidates = [x for x in range(NP) if x != i]
                offspring = copy.deepcopy(P[i])

                p = random.uniform(0, 1)
                idxOfBest = random.randint(0, math.floor(p * NP))
                r1, r2 = random.sample(candidates, 2)
                zipped = zip(P, results)
                xbest = sorted(zipped, key=lambda x: x[1])[idxOfBest][0]
                x1 = P[r1]
                x2 = P[r2]

                v = P[i] + F * (xbest - P[i]) + F * (x1 - x2)
                
                jrand = random.randint(0, dimension-1)
                for j in range(dimension):
                    rand = random.uniform(0, 1)
                    if rand < CR or j == jrand:
                        offspring[j] = self.checkBoundary(v[j])

                value = self.f.evaluate(func_num, offspring)
                if value == "ReachFunctionLimit":
                    break
                value = float(value)
                # print(values)
                if value < results[i]:
                    P[i] = offspring
                    results[i] = value
                    Scr.append(CR)
                    Sf.append(F)

                self.eval_times += 1

            c = random.uniform(0, 1)
            if len(Scr) > 0:
                meanCR = (1-c) * meanCR + c * sum(Scr)/len(Scr)
            if sum(Sf) != 0:
                meanF = (1-c) * meanF + c * sum([f**2 for f in Sf])/sum(Sf)

            self.optimal_value = min(results)
            self.optimal_solution = P[np.argmin(results)]

    def CoDE(self, maxFES, NP=6):
        dimension = self.dim
        # F pool and CR pool
        F = [1, 1, 0.8]
        CR = [0.1, 0.9, 0.2]

        # Randomly generate initial population
        P = np.random.uniform(low=self.lower, high=self.upper, size=(NP, dimension))

        # Evaluate objective function
        results = []

        for i in range(NP):
            results.append(float(self.f.evaluate(func_num, P[i])))

        self.eval_times = NP
        while self.eval_times < maxFES:
            print('=====================FE=====================')

            for i in range(NP):
                # Generate three offsprings with mutation
                candidates = [x for x in range(NP) if x != i]
                offsprings = [copy.deepcopy(P[i]), copy.deepcopy(P[i]), np.zeros(dimension)]

                r1, r2, r3 = random.sample(candidates, 3)
                control = random.randint(0, 2)
                jrand = random.randint(0, dimension-1)
                for j in range(dimension):
                    rand = random.uniform(0, 1)
                    if rand <= CR[control] or j == jrand:
                        tmp = P[r1][j] + F[control] * (P[r2][j] - P[r3][j])
                        offsprings[0][j] = self.checkBoundary(tmp)

                r1, r2, r3, r4, r5 = random.sample(candidates, 5)
                control = random.randint(0, 2)
                jrand = random.randint(0, dimension-1)
                for j in range(dimension):
                    rand = random.uniform(0,1)
                    if rand <= CR[control] or j == jrand:
                        tmp = P[r1][j] + F[control] * (P[r2][j] - P[r3][j]) + F[control] * (P[r4][j] - P[r5][j])
                        offsprings[1][j] = self.checkBoundary(tmp)

                r1, r2, r3 = random.sample(candidates, 3)
                control = random.randint(0, 2)
                rand = random.uniform(0,1) # uniformly distributed random number between 0 and 1
                for j in range(dimension):
                    offsprings[2][j] = self.checkBoundary(P[i][j] + rand * (P[r1][j] - P[i][j]) + F[control] * (P[r2][j] - P[r3][j]))
                
                # Choose the best trial vector from the three trial vectors 
                values = []
                for j in range(3):
                    value = self.f.evaluate(func_num, offsprings[j])
                    if value == "ReachFunctionLimit":
                        break
                    values.append(float(value))

                if self.eval_times > maxFES:
                    break

                u = min(values)
                if u < results[i]:
                    P[i] = offsprings[np.argmin(values)]
                    results[i] = u
               
                self.eval_times += 3

            self.optimal_value = min(results)
            self.optimal_solution = P[np.argmin(results)]
            print("Best solution in the population: {}\n".format(self.get_optimal()[1]))

    def RandomSearch(self, maxFES):
        while self.eval_times < maxFES:
            print('=====================FE=====================')
            print(self.eval_times)

            solution = np.random.uniform(np.full(self.dim, self.lower), np.full(self.dim, self.upper), self.dim)
                
            print(solution)
            value = self.f.evaluate(func_num, solution)
            self.eval_times += 1

            if value == "ReachFunctionLimit":
                print("ReachFunctionLimit")
                break            
            if float(value) < self.optimal_value:
                self.optimal_solution[:] = solution
                self.optimal_value = float(value)

            print("optimal: {}\n".format(self.get_optimal()[1]))

    def run(self, FES): # main part for your implementation
        print('lower:', self.lower)
        print('upper:', self.upper)
        print('dimension:', self.dim)
        print('FES:', FES)
        # self.RandomSearch(maxFES=FES)
        # self.CoDE(maxFES=FES)
        self.JADE(maxFES = FES)

if __name__ == '__main__':
    func_num = 1

    fes = 0
    #function1: 1000, function2: 1500, function3: 2000, function4: 2500

    while func_num < 5:
        if func_num == 1:
            fes = 1000
        elif func_num == 2:
            fes = 1500
        elif func_num == 3:
            fes = 2000 
        else:
            fes = 2500

        # you should implement your optimizer
        op = RS_optimizer(func_num)
        op.run(fes)
        
        best_input, best_value = op.get_optimal()

        print(best_input, best_value)
        
        # change the name of this file to your student_ID and it will output properlly
        with open("{}_function{}.txt".format(__file__.split('_')[0], func_num), 'w+') as f:
            for i in range(op.dim):
                f.write("{}\n".format(best_input[i]))
            f.write("{}\n".format(best_value))
        func_num += 1 
