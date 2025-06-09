from typing import Any, Dict, Tuple
from joblib import dump
from math import sqrt, inf
from matplotlib import pyplot as plt
import time

FILENAME = "att48.tsp"

def main():
    
    nodes: Dict[int, Tuple[float, float]] = readNodes(FILENAME)    
    dimension: int = len(nodes)
    edges = calculateEdges(nodes, dimension)
    start = 1
    
    startTime = time.perf_counter()
    firstSolution, firstDistance = calculateNearestNeighbour(edges, dimension, start)
    endTime = time.perf_counter()
    totalTime = endTime-startTime
    bestSolution, bestDistance, bestStart = calculateBestNearestNeighbour(edges, dimension)
    plot_solution(nodes, firstSolution, title=f"Soluzione Greedy ATSP con start: {start}. Distance: {firstDistance:.2f}. Time: {totalTime:.6f}")
    plot_solution(nodes, bestSolution, title=f"Soluzione Greedy ATSP multistart. Best start: {bestStart}. Distance: {bestDistance:.2f}")
    dump({"Solution": firstSolution, "Distance": firstDistance, "Time": totalTime}, "firstSolutionGreedy.joblib")
    dump({"Solution": bestSolution, "Distance": bestDistance}, "bestSolutionGreedy.joblib")

    return

def readNodes(filename: str) -> Dict[Any, Tuple[float, float]]:
    
    nodes = {}
    
    # Lettura nodi e coordinate
    with open(filename, mode="r") as graph:
        
        line = graph.readline()
        
        while not line.startswith("EOF"):
            
            if line.startswith("DIMENSION"):
                dimension = int(line.split()[1])
            
            if line.startswith("NODE"):
                for i in range(dimension):
                    
                    line = graph.readline()
                    splitLine = line.split()
                    nodes[int(splitLine[0])] = (float(splitLine[1]), float(splitLine[2]))
            
            line = graph.readline()
        
        graph.close()
    
    return nodes

def calculateEdges(nodes: Dict[int, Tuple[float, float]], dimension: int) -> Dict[Tuple[int, int], float]:
    
    distances = {}
    
    for i in range(1, dimension + 1):
        for j in range(1, dimension + 1):
            if i != j:
                xDistance = nodes.get(i)[0] - nodes.get(j)[0]
                yDistance = nodes.get(i)[1] - nodes.get(j)[1]
                distances[(i, j)] = sqrt(xDistance**2 + yDistance**2)
    
    return distances

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
    
def plot_solution(nodes, solution, title="Soluzione Greedy TSP"):
    
    plt.figure(figsize=(10, 8))
    
    for (i,j) in solution:
        x = [nodes[i][0], nodes[j][0]]
        y = [nodes[i][1], nodes[j][1]]
        plt.plot(x, y, 'b-', linewidth=1)
    
    for k, (x,y) in nodes.items():
        plt.plot(x, y, 'ro')
        plt.text(x, y + 1, str(k), fontsize=8, ha='center')
    
    plt.title(title)
    plt.axis('equal')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()