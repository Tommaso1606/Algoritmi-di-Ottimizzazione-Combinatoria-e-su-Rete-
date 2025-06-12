import joblib as jb
from common import readNodes, calculateEdges, plotOrientedSolution,buildTour,twoOptSwap,valueObj
from typing import List, Tuple, Dict, Any
import time
import copy
from itertools import combinations
import random

FILE_NAME = "att48/att48.tsp"

def main():
    nodes = readNodes(FILE_NAME)    
    dimension = len(nodes)
    distances = calculateEdges(nodes, dimension)
    
    sol = jb.load("att48/solutions/att48_greedy0.joblib")
    solutionStart = sol.get('solution')
    
    tour = buildTour(solutionStart)
    #print(f'tour: {tour}')
    startTime = time.perf_counter()
    #tour = localSearchFirstImprovement(tour,distances)
    tour = localSearchBestImprovement(tour,distances)
    endTime = time.perf_counter()
    
    solution = [(tour[i], tour[i+1]) for i in range(len(tour) - 1)]
    
    #plotOrientedSolution(nodes, solution, title=f"Soluzione Local Search First Improvement ATSP. Distance: {valueObj(tour,distances):.2f}. Time: {(endTime-startTime):.6f}s")
    plotOrientedSolution(nodes, solution, title=f"Soluzione Local Search Best Improvement ATSP. Distance: {valueObj(tour,distances):.2f}. Time: {(endTime-startTime):.6f}s")
    
def localSearchFirstImprovement(tour: List, distances: Dict[Tuple[int,int], float]) -> List:

    moveSpace = generateTwoOptSpace(tour)

    noImprovement = False

    while not noImprovement:
        improvement_found = False
        for (i, k) in moveSpace:
            newTour = twoOptSwap(tour, i, k)
            if isAcceptable(tour, newTour, distances):
                tour = newTour
                moveSpace = generateTwoOptSpace(tour)
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
                #print(f'miglioramento trovato {valueObj(tour,distances)}')
                improvement_found = True
        if not improvement_found:
            noImprovement = True
        else: 
            moveSpace = generateTwoOptSpace(tour)

    return tour

def isAcceptable(startTour: List[int], endTour: List[int],distances: Dict[Tuple[int, int], float]) -> bool:
    return valueObj(endTour, distances)<valueObj(startTour, distances)

def generateTwoOptSpace(tour: List[int]) -> List[Tuple[int,int]]:
    n = len(tour)
    swaps = list(combinations(range(1,n-2), 2))
    random.seed(42)
    random.shuffle(swaps)
    return swaps

if __name__=="__main__":
    main()
