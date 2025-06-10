from typing import List, Tuple,Dict,Any
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


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

def plotSolution(Nodes: Dict[int, Tuple[float, float]], edges, title="Soluzione TSP"):
    plt.figure(figsize=(50, 50))
    #ax = plt.gca()  # Per aggiungere oggetti disegnabili (patches)
    for (i, j) in edges:
        x = [Nodes[i][0], Nodes[j][0]]
        y = [Nodes[i][1], Nodes[j][1]]
        plt.plot(x, y, 'b-', linewidth=1)
        
        #x_start, y_start = Nodes[i]
        #x_end, y_end = Nodes[j]
        #arrow = FancyArrowPatch(
        #    (x_start, y_start), (x_end, y_end),
        #    arrowstyle='->',
        #    color='blue',
        #    mutation_scale=10,
        #    linewidth=1
        #)
        #ax.add_patch(arrow)
    for k, (x, y) in Nodes.items():
        plt.plot(x, y, 'ro')
        plt.text(x, y + 1, str(k), fontsize=8, ha='center')
    plt.title(title)
    plt.axis('equal')
    plt.grid(True)
    plt.show()