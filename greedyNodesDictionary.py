from typing import Any, Dict, Tuple
from joblib import dump
from math import sqrt, inf
from matplotlib import pyplot as plt
import time
from common import calculateEdges, plotOrientedSolution,readNodes


ISTANCE_NAME = "att48"

def main():
    
    nodes: Dict[int, Tuple[float, float]] = readNodes(f"{ISTANCE_NAME}/{ISTANCE_NAME}.tsp")    
    dimension: int = len(nodes)
    edges = calculateEdges(nodes, dimension)
    start = 1
    
    startTime = time.perf_counter()
    firstSolution, firstDistance = calculateNearestNeighbour(edges, dimension, start)
    endTime = time.perf_counter()
    totalTime = endTime-startTime
    bestSolution, bestDistance, bestStart = calculateBestNearestNeighbour(edges, dimension)
    plotOrientedSolution(nodes, firstSolution, title=f"Soluzione Greedy ATSP con start: {start}. Distance: {firstDistance:.2f}. Time: {totalTime:.6f}")
    plotOrientedSolution(nodes, bestSolution, title=f"Soluzione Greedy ATSP multistart. Best start: {bestStart}. Distance: {bestDistance:.2f}")
    dump({"Solution": firstSolution, "Distance": firstDistance, "Time": totalTime}, "firstSolutionGreedy.joblib")
    dump({"Solution": bestSolution, "Distance": bestDistance}, "bestSolutionGreedy.joblib")

    return

def calculateNearestNeighbour(edges: Dict[Tuple[int, int], float], dimension: int, start: int = 1) -> Tuple[Dict[Tuple[int, int], float], float]:

    solution: Dict[Tuple[int, int], float] = {}
    distance: float = 0
    
    currentNode: int = start
    visitedNodes = set()
    
    for _ in range(0, dimension - 1):
        
        # Scelta di costo minimo
        choices = ((key, value) for key, value in edges.items() if key[0] == currentNode and key[1] not in visitedNodes)
        minimum = min(choices, key=lambda choices: choices[1])
    
        solution[minimum[0]] = minimum[1]
        distance += minimum[1]
        
        visitedNodes.add(currentNode)
        currentNode = minimum[0][1]
    
    # Last move
    solution[(currentNode, start)] = edges.get((currentNode, start))
    distance += edges.get((currentNode, start))
    
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