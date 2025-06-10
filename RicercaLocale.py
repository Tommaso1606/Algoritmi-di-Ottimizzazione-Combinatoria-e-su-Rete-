import joblib as jb
from common import readNodes, calculateEdges, plotSolution
from typing import List, Tuple, Dict, Any
import time
import copy
from itertools import combinations
import random

FILE_NAME = "ch130.tsp"

def main():
    nodes = readNodes(FILE_NAME)
    dimension = len(nodes)
    distances = calculateEdges(nodes, dimension)
    
    sol = jb.load("bestSolutionGreedyMatrixCh130.joblib")
    solutionStart = sol.get('Solution')
    
    tour = buildTour(solutionStart)
    #print(f'tour: {tour}')
    startTime = time.perf_counter()
    tour = localSearchFirstImprovement(tour,distances)
    #tour = localSearchBestImprovement(tour,distances)
    endTime = time.perf_counter()
    
    solution = [(tour[i], tour[i+1]) for i in range(len(tour) - 1)]
    
    plotSolution(nodes, solution, title=f"Local Search First Improvement TSP con distance: {valueObj(tour,distances):.2f}. Tempo: {(endTime-startTime):.6f} s.")
    #plotSolution(nodes, solution, title=f"Local Search Best Improvement TSP con distance: {valueObj(tour,distances):.2f}. Tempo: {(endTime-startTime):.6f} s.")
    
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

def buildTour(edges: List[Tuple[int,int]]) -> List:
    edgeSorted = sorted(edges, key=lambda x: x[0])
    
    Tour: list = []
    
    startNode = edgeSorted[0][0]
    
    Tour.append(startNode)
    
    currentNode = startNode
    
    while True: 
        nextEdge = None
        for edge in edgeSorted:
            if edge[0] == currentNode:
                nextEdge = edge
                break
        
        nextNode = nextEdge[1]
        
        if nextNode == startNode:
            Tour.append(startNode)
            break
        
        Tour.append(nextNode)
        currentNode = nextNode

    return Tour
    
def valueObj(Tour: list, Distances: Dict[Tuple[int,int],float]) -> float:
    return sum(Distances[(Tour[i], Tour[i+1])] for i in range(len(Tour)-1))

def twoOptSwap(Tour: List, i: int, k: int) -> List:
    return Tour[:i] + list(reversed(Tour[i:k+1])) + Tour[k+1:]

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
