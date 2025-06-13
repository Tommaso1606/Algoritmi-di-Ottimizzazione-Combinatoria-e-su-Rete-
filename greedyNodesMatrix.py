from typing import Any, Dict, List, Tuple
from joblib import dump
from math import sqrt, inf
from matplotlib import pyplot as plt
from common import readNodes,plotOrientedSolution
import time

import os
import shutil

PATH = "ch130.tsp"
BEST = 0
SOLUTION = "solution"
DISTANCE = "distance"
START = "start"
TIME = "time"

def main():
    
    # Crea le cartelle necessarie a salvare le soluzioni
    filename: str = PATH.split(".")[0]
    tspFolder: str = "./" + filename
    solutionsPath = tspFolder + "/solutions"
    os.makedirs(solutionsPath, exist_ok=True)
    
    if os.path.exists(f"./{PATH}"):
        shutil.move(f"./{PATH}", tspFolder)
    
    # Calcolo di nodi, dimensione grafo e archi
    nodes: Dict[int, Tuple[float, float]] = readNodes(tspFolder + f"/{PATH}")
    dimension: int = len(nodes)
    edges = calculateEdges(nodes, dimension)
    
    solutions: Dict[int, Dict[str, Any]] = {}
    
    # Miglior Greedy
    startTime = time.perf_counter()
    solution, distance, start = calculateBestNearestNeighbour(edges, dimension)
    endTime = time.perf_counter()
    
    solutions[BEST] = {
        SOLUTION: solution,
        DISTANCE: distance,
        START: start,
        TIME: endTime - startTime
    }
    
    # Una sola iterazione con start = 1
    start = 1
    startTime = time.perf_counter()
    solution, distance = calculateNearestNeighbour(edges, dimension, start)
    endTime = time.perf_counter()
    
    solutions[start] = {
        SOLUTION: solution,
        DISTANCE: distance,
        START: start,
        TIME: endTime - startTime
    }
    
    for key in solutions.keys():
        solution = solutions[key]
        plotOrientedSolution(nodes, solution[SOLUTION], title=f"Soluzione Greedy ATSP con start: {solution[START]}. Distance: {solution[DISTANCE]:.2f}. Time: {solution[TIME]:.6f}s",istanceName=PATH[:-4],fileName=f"{PATH[:-4]}_Greedy_{solution[START]}")
        dump({SOLUTION: solution[SOLUTION], START: solution[START], DISTANCE: solution[DISTANCE], TIME: solution[TIME]}, f"./{solutionsPath}/{filename}_greedy{key}.joblib")
    return

def calculateEdges(nodes: Dict[int, Tuple[float, float]], dimension: int):
    
    distancesMatrix: List[List[float]] = [[0.0 for _ in range(dimension)] for _ in range(dimension)]
    
    for i in range(0, dimension):
        for j in range(0, dimension):
            if i != j:
                xDistance = nodes.get(i+1)[0] - nodes.get(j+1)[0]
                yDistance = nodes.get(i+1)[1] - nodes.get(j+1)[1]
                distancesMatrix[i][j] = sqrt(xDistance**2 + yDistance**2)
            else:
                distancesMatrix[i][j] = inf
    
    return distancesMatrix

def calculateNearestNeighbour(edges: List[List[int]], dimension: int, start: int = 1):

    solution: Dict[Tuple[int, int], float] = {}
    distance: float = 0.0
    
    currentNode: int = start - 1
    visitedNodes = set()
    visitedNodes.add(currentNode)
    
    for _ in range(0, dimension - 1):
        minimum: float = inf
        minNode: int = -1
        
        for j in range(dimension):
            if (j not in visitedNodes) and (edges[currentNode][j] < minimum):
                minimum = edges[currentNode][j]
                minNode = j
                
        solution[(currentNode+1, minNode+1)] = minimum
        distance += minimum
        
        visitedNodes.add(minNode)
        currentNode = minNode
    
    # Ultima mossa
    solution[(currentNode+1, start)] = edges[currentNode][start-1]
    distance += edges[currentNode][start-1]
    
    return solution, distance

def calculateBestNearestNeighbour(edges: Dict[Tuple[int, int], float], dimension: int) -> Tuple[Dict[Tuple[int, int], float], float, int]:
    
    solution: Dict[Tuple[int, int], float] = {}
    distance: float = inf
    bestStart: int = 0
    
    for i in range(1, dimension + 1):
        
        tempSolution, tempDistance = calculateNearestNeighbour(edges, dimension, i)
        
        if tempDistance < distance:
            distance = tempDistance
            solution = tempSolution
            bestStart = i
    
    return solution, distance, bestStart

if __name__ == "__main__":
    main()