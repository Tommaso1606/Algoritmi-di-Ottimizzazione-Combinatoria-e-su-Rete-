from typing import List, Tuple,Dict,Any
from math import sqrt
import matplotlib.pyplot as plt

def readNodes(filename: str) -> Dict[int, Tuple[float, float]]:
   
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

def plotOrientedSolution(nodes, solution, title="Soluzione Greedy TSP"):
    
    plt.figure(figsize=(10, 8))
    
    for (i,j) in solution:
        x_start, y_start = nodes[i]
        x_end, y_end = nodes[j]
        
        # Disegna una freccia orientata da i a j
        plt.annotate("", xy=(x_end, y_end), xytext=(x_start, y_start), arrowprops=dict(arrowstyle="->", color="blue", lw=1))
    
    for k, (x,y) in nodes.items():
        plt.plot(x, y, 'ro')
        plt.text(x, y + 1, str(k), fontsize=8, ha='center')
    
    plt.title(title)
    plt.axis('equal')
    plt.grid(True)
    plt.show()
    
def valueObj(tour: list[int], distances: Dict[Tuple[int,int],float]) -> float:
    return sum(distances[(tour[i], tour[i+1])] for i in range(len(tour)-1))

def twoOptSwap(tour: List[int], i: int, k: int) -> List:
    return tour[:i] + list(reversed(tour[i:k+1])) + tour[k+1:]

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