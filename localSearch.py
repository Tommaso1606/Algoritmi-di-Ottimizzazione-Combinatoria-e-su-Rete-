import joblib as jb
from common import readNodes, calculateEdges, plotOrientedSolution,buildTour,twoOptSwap,valueObj
from typing import List, Tuple, Dict, Any
import time
import copy
from itertools import combinations
import random

ISTANCE_NAME = "ch130"

def main():
    nodes = readNodes(f"{ISTANCE_NAME}/{ISTANCE_NAME}.tsp")
    dimension = len(nodes)
    distances = calculateEdges(nodes, dimension)
    
    sol = jb.load(f"{ISTANCE_NAME}/solutions/{ISTANCE_NAME}_greedy0.joblib")
    solutionStart = sol.get('solution')
    solutionTime = sol.get('time')
    
    tour = buildTour(solutionStart)

    # Soluzione First Improvement
    startTime = time.perf_counter()
    tour1 = localSearchFirstImprovement(tour,distances)
    endTime = time.perf_counter()
    endTime += solutionTime
    
    solution = [(tour1[i], tour1[i+1]) for i in range(len(tour1) - 1)]
    
    plotOrientedSolution(nodes, solution, title=f"Soluzione Local Search First Improvement ATSP. Distance: {valueObj(tour1,distances):.2f}. Time: {(endTime-startTime):.6f}s",istanceName=ISTANCE_NAME,fileName=f"{ISTANCE_NAME}_LS_FI")
    
    # Soluzione Best Improvement
    startTime = time.perf_counter()
    tour2 = localSearchBestImprovement(tour,distances)
    endTime = time.perf_counter()
    endTime += solutionTime
    
    solution = [(tour2[i], tour2[i+1]) for i in range(len(tour2) - 1)]
    
    plotOrientedSolution(nodes, solution, title=f"Soluzione Local Search Best Improvement ATSP. Distance: {valueObj(tour2,distances):.2f}. Time: {(endTime-startTime):.6f}s",istanceName=ISTANCE_NAME,fileName=f"{ISTANCE_NAME}_LS_BI")
    
def localSearchFirstImprovement(tour: List, distances: Dict[Tuple[int,int], float]) -> List:

    moveSpace = generateTwoOptSpace(tour)

    noImprovement = False

    while not noImprovement:
        improvement_found = False
        for (i, k) in moveSpace:
            newTour = twoOptSwap(tour, i, k)
            if isAcceptable(tour, newTour, distances):
                tour = newTour
                #moveSpace = generateTwoOptSpace(tour)
                random.shuffle(moveSpace)
                improvement_found = True
                break
        if not improvement_found:
            noImprovement = True

    return tour

def localSearchBestImprovement(tour: List, distances: Dict[Tuple[int,int], float]) -> List:

    moveSpace = generateTwoOptSpace(tour)

    noImprovement = False

    while not noImprovement:
        improvement_found = False
        for (i, k) in moveSpace:
            newTour = twoOptSwap(tour, i, k)
            if isAcceptable(tour, newTour, distances):
                tour = newTour
                improvement_found = True
        if not improvement_found:
            noImprovement = True
        #else: 
            #moveSpace = generateTwoOptSpace(tour)
            #random.shuffle(moveSpace)

    return tour

def isAcceptable(startTour: List[int], endTour: List[int],distances: Dict[Tuple[int, int], float]) -> bool:
    return valueObj(endTour, distances)<valueObj(startTour, distances)

def generateTwoOptSpace(tour: List[int]) -> List[Tuple[int,int]]:
    n = len(tour)
    # Combinazione di indici
    swaps = list(combinations(range(1,n-2), 2))
    #random.seed(42)
    #random.shuffle(swaps) va fatto nella funzione chiamante
    return swaps

if __name__=="__main__":
    main()