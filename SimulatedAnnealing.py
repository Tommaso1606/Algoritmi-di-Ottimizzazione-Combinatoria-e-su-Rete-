import math
import random
import joblib as jb
from common import readNodes, calculateEdges, plotOrientedSolution,buildTour,twoOptSwap,valueObj
from typing import List, Tuple, Dict, Any
import time
import copy
from itertools import combinations
import itertools

FILE_NAME = "lin318/lin318.tsp"

def main():
    # Da cambiare in caso di grid search
    #random.seed(592)


    nodes = readNodes(FILE_NAME)
    dimension = len(nodes)
    distances = calculateEdges(nodes, dimension)
    
    sol = jb.load("lin318/solutions/lin318_greedy0.joblib")
    solutionStart = sol.get('solution')
    
    tour = buildTour(solutionStart)
    
    # Iperparametri
    tK = 90 # Temperatura iniziale
    nIter = 40 # Numero iterazioni per ogni temperatura
    alfa = 0.995 # Coefficiente di raffredamento
    nNoImprovement = 4000 # Numero di mosse peggiorative consecutive
    
    startTime = time.perf_counter()
    #tour = simulatedAnnealing(tK,alfa,tour,distances,nIter,nNoImprovement)
    endTime = time.perf_counter()
    
    gridSearchSimulatedAnnealing(tour, distances)
    
    #solution = [(tour[i], tour[i+1]) for i in range(len(tour) - 1)]
    
    #plotOrientedSolution(nodes, solution, title=f"Soluzione Simulated Annealing ATSP. Distance: {valueObj(tour,distances):.2f}. Time: {(endTime-startTime):.6f}s")
    
def gridSearchSimulatedAnnealing(startTour: List[int], distances: Dict[Tuple[int, int], float]):
    tK_values = [100, 150, 90, 80]
    alfa_values = [0.99, 0.995, 0.98, 0.97]
    nIter_values = [40, 50, 20, 30, 10]
    nNoImprovement_values = [1500,2000,2500,3000,4000,5000]

    param_grid = list(itertools.product(tK_values, alfa_values, nIter_values, nNoImprovement_values))

    best_score = float("inf")
    best_params = None
    best_seed = None

    print(f"Testing {len(param_grid)} combinations...")

    startTime = time.perf_counter()

    for idx, (tK, alfa, nIter, nNoImprovement) in enumerate(param_grid):
        seed = 42 + idx  # seed unico per ogni combinazione, riproducibile

        print(f"\nTesting: tK={tK}, alfa={alfa}, nIter={nIter}, nNoImprovement={nNoImprovement}, seed={seed}")

        tour_copy = copy.deepcopy(startTour)
        resultTour = simulatedAnnealing(tK, alfa, tour_copy, distances, nIter, nNoImprovement, seed=seed)
        score = valueObj(resultTour, distances)

        print(f"→ Score: {score:.2f}")

        if score < best_score:
            best_score = score
            best_params = (tK, alfa, nIter, nNoImprovement)
            best_seed = seed

    endTime = time.perf_counter()
    
    print("\n🏆 Best combination:")
    print(f"tK={best_params[0]}, alfa={best_params[1]}, nIter={best_params[2]}, nNoImprovement={best_params[3]}, seed={best_seed}")
    print(f"Best score: {best_score:.2f}. Time: {(endTime-startTime):.6f}s")

def simulatedAnnealing(T_k: float, alfa: float, tour: List[int], distances: Dict[Tuple[int,int], float], nIter: int, nNoImprovement: int, seed: int = None) -> List[int]:
    
    if seed is not None:
        random.seed(seed)

    noImprovement: int = 0
    bestTour: List[int] = copy.deepcopy(tour)
    
    while T_k > 1e-3 and noImprovement < nNoImprovement:
        for _ in range(nIter):
            i, k = sorted(random.sample(range(1, len(tour) - 2), 2))
            newTour = twoOptSwap(tour, i, k)
            
            if isAcceptable(tour, newTour, T_k, distances):
                tour = newTour        
                if valueObj(tour, distances) < valueObj(bestTour, distances):
                    bestTour = copy.deepcopy(tour)
                    noImprovement = 0
                else:
                    noImprovement += 1    
            else:
                noImprovement += 1    
        T_k *= alfa

    return bestTour

def isAcceptable(startTour: List[int], endTour: List[int], T_k: float, distances: Dict[Tuple[int, int], float]) -> bool:
    delta = valueObj(endTour, distances) - valueObj(startTour, distances)
    if delta < 0:
        return True
    else:
        p = math.exp(-delta / T_k)
        return random.random() < p

if __name__=="__main__":
    main()