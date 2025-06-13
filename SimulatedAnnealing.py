import math
import random
import joblib as jb
from common import readNodes, calculateEdges, plotOrientedSolution,buildTour,twoOptSwap,valueObj
from typing import List, Tuple, Dict, Any
import time
import copy
from itertools import combinations
import itertools
import json

ISTANCE_NAME = "ch130"

def main():
    nodes = readNodes(f"{ISTANCE_NAME}/{ISTANCE_NAME}.tsp")
    dimension = len(nodes)
    distances = calculateEdges(nodes, dimension)
    
    sol = jb.load(f"{ISTANCE_NAME}/solutions/{ISTANCE_NAME}_greedy0.joblib")
    solutionStart = sol.get('solution')
    solutionDistance = sol.get('distance')
    solutionTime = sol.get('time')
    tour = buildTour(solutionStart)
    n = len(tour)
    
    # Iterazione con parametri predefiniti
    tK = 10*abs(solutionDistance/2) # Temperatura iniziale
    tF = 1e-4*abs(solutionDistance) # Temperatura finale
    nIter = n # Numero iterazioni per ogni temperatura
    alfa = 0.995 # Coefficiente di raffredamento
    nNoImprovement = 5*n # Numero di mosse peggiorative consecutive

    print(f"Parametri iniziali: tK={tK}, alfa={alfa}, nIter={nIter}, nNoImprovement={nNoImprovement}, tf={tF}")

    startTime = time.perf_counter()
    tour1 = simulatedAnnealing(tK,alfa,tour,distances,nIter,nNoImprovement,tF)
    endTime = time.perf_counter()
    endTime += solutionTime
    
    solution = [(tour1[i], tour1[i+1]) for i in range(len(tour1) - 1)]
    
    plotOrientedSolution(nodes, solution, title=f"Soluzione Simulated Annealing ATSP Default Params. Distance: {valueObj(tour1,distances):.2f}. Time: {(endTime-startTime):.6f}s,",istanceName=ISTANCE_NAME,fileName=f"{ISTANCE_NAME}_SA_DP")


    # Iterazione con parametri ricercati attraverso RandomSearch
    nTest = 1000

    startTime = time.perf_counter()
    tour2 = randomSearchSimulatedAnnealing(tour, distances,tK,tF,nIter,nNoImprovement,alfa,nTest)
    endTime = time.perf_counter()
    endTime += solutionTime
    
    solution = [(tour2[i], tour2[i+1]) for i in range(len(tour2) - 1)]
    
    plotOrientedSolution(nodes, solution, title=f"Soluzione Simulated Annealing ATSP Random Search Params. Distance: {valueObj(tour2,distances):.2f}. Time: {(endTime-startTime):.6f}s",istanceName=ISTANCE_NAME,fileName=f"{ISTANCE_NAME}_SA_RS")

    
    # Iterazione con parametri ricercati attraverso GridSearch
    paramGrid: List = []
    
    with open('ParamsStat.json', 'r') as f:
        params = json.load(f)

    tK_values = params['tK_values']
    alfa_values = params['alfa_values']
    nIter_values = params['nIter_values']
    nNoImprovement_values = params['nNoImprovement_values']
    tF_values = params['tF_values']
    
    paramGrid = list(itertools.product(tK_values, alfa_values, nIter_values, nNoImprovement_values,tF_values))
    
    startTime = time.perf_counter()
    tour3 = gridSearchSimulatedAnnealing(tour, distances,paramGrid)
    endTime = time.perf_counter()
    endTime += solutionTime
    
    solution = [(tour3[i], tour3[i+1]) for i in range(len(tour3) - 1)]
    
    plotOrientedSolution(nodes, solution, title=f"Soluzione Simulated Annealing ATSP Grid Search Params. Distance: {valueObj(tour3,distances):.2f}. Time: {(endTime-startTime):.6f}s",istanceName=ISTANCE_NAME,fileName=f"{ISTANCE_NAME}_SA_GS")
       
def gridSearchSimulatedAnnealing(startTour: List[int], distances: Dict[Tuple[int, int], float],paramGrid: List[Tuple[float,float,float,float,float]]):

    best_score = float("inf")
    best_params = None
    best_seed = None
    best_tour: List = []

    print(f"Testing {len(paramGrid)} combinations...")

    startTime = time.perf_counter()

    base_seed = int(time.time())

    for idx, (tK, alfa, nIter, nNoImprovement,tF) in enumerate(paramGrid):
        seed = base_seed + idx  # seed unico per ogni combinazione, riproducibile

        print(f"\nTesting: tK={tK}, alfa={alfa}, nIter={int(nIter)}, nNoImprovement={int(nNoImprovement)},tf={tF} ,seed={seed}")

        tour_copy = copy.deepcopy(startTour)
        resultTour = simulatedAnnealing(tK, alfa, tour_copy, distances, int(nIter), int(nNoImprovement), tF, seed=seed)
        score = valueObj(resultTour, distances)

        print(f"→ Score: {score:.2f}")

        if score < best_score:
            best_score = score
            best_params = (tK, alfa, nIter, nNoImprovement)
            best_seed = seed
            best_tour = resultTour

    endTime = time.perf_counter()
    
    print("\n🏆 Best combination:")
    print(f"tK={best_params[0]}, alfa={best_params[1]}, nIter={best_params[2]}, nNoImprovement={best_params[3]}, tf={tF}, seed={best_seed}")
    print(f"Best score: {best_score:.2f}. Time: {(endTime-startTime):.6f}s")
    
    return best_tour

def randomSearchSimulatedAnnealing(startTour: List[int], distances: Dict[Tuple[int, int], float],tK: float,tF: float,nIter: int,nNoImprovement: int,alfa: float,nTest: int):

    best_score = float("inf")
    best_params = None
    best_seed = None
    best_tour: List = []

    print(f"Testing {nTest} combinations...")

    startTime = time.perf_counter()
    
    base_seed = int(time.time())

    for idx in range(nTest):
        seed = base_seed + idx  # seed unico per ogni combinazione, riproducibile

        alfaValue = random.uniform(0.8,alfa)
        nIterValue = random.randint(10,int(1.5*nIter))
        tKValue = random.uniform(0.25*tK,1.5*tK)
        tFValue = random.uniform(0.25*tF,1.5*tF)
        nNoImprovementValue = random.randint(10,int(1.5*nNoImprovement))

        print(f"\nTesting: tK={tKValue}, alfa={alfaValue}, nIter={nIterValue}, nNoImprovement={nNoImprovementValue},tf={tFValue} ,seed={seed}")

        tour_copy = copy.deepcopy(startTour)
        resultTour = simulatedAnnealing(tKValue, alfaValue, tour_copy, distances, nIterValue, nNoImprovementValue, tFValue, seed=seed)
        score = valueObj(resultTour, distances)

        print(f"→ Score: {score:.2f}")

        if score < best_score:
            best_score = score
            best_params = (tKValue, alfaValue, nIterValue, nNoImprovementValue,tFValue)
            best_seed = seed
            best_tour = resultTour

    endTime = time.perf_counter()
    
    print("\n🏆 Best combination:")
    print(f"tK={best_params[0]}, alfa={best_params[1]}, nIter={best_params[2]}, nNoImprovement={best_params[3]}, tf={best_params[4]}, seed={best_seed}")
    print(f"Best score: {best_score:.2f}. Time: {(endTime-startTime):.6f}s")
    
    return best_tour

def simulatedAnnealing(T_k: float, alfa: float, tour: List[int], distances: Dict[Tuple[int,int], float], nIter: int, nNoImprovement: int, tF: float, seed: int = int(time.time())) -> List[int]:
    
    if seed is not None:
        random.seed(seed)

    noImprovement: int = 0
    bestTour: List[int] = copy.deepcopy(tour)
    
    while T_k > tF and noImprovement < nNoImprovement:
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