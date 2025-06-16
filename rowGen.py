from common import readNodes,calculateEdges,plotOrientedSolution
import gurobipy as gp
from gurobipy import GRB
from typing import List, Tuple,Dict,Any
import time

ISTANCE_NAME = "ch130"

def main():

    nodes = readNodes(f"{ISTANCE_NAME}/{ISTANCE_NAME}.tsp")
    print(nodes)

    dimension = len(nodes)

    dist = calculateEdges(nodes,dimension)

    startTime = time.perf_counter()
    solution,distance = solve(nodes,dist)
    endTime = time.perf_counter()  
      
    plotOrientedSolution(nodes, solution, title=f"Soluzione Ottima ATSP. Distance: {distance:.2f}. Time: {(endTime-startTime):.6f}s",istanceName=ISTANCE_NAME,fileName=f"{ISTANCE_NAME}_Ottimo")

def solve(nodes: Dict[int, Tuple[float, float]], dist: Dict[Tuple[int, int], float]) -> Tuple[List[Tuple[int,int]],float]:
    
    mod = gp.Model("TSP")

    Xvars = mod.addVars(dist.keys(), obj = dist, vtype = GRB.BINARY, name = "x")

    # Outstar
    mod.addConstrs(gp.quicksum(Xvars[i,j] for j in nodes if i != j) == 1 for i in nodes)

    # Instar
    mod.addConstrs(gp.quicksum(Xvars[j,i] for j in nodes if j != i) == 1 for i in nodes)
    #mod.addConstrs(Xvars.sum('*',i) == 1 for i in Nodes)
    
    # Disattiva l'output nel terminale di Gurobi
    #mod.setParam("OutputFlag", 0)

    stop = False
    
    nItr = 0

    distance = 0
    
    while not stop :
    
        nItr = nItr +1 

        mod.optimize()

        # Lower Bound
        mod.ObjVal
        
        distance = mod.ObjVal

        solution = []
        for (i,j) in Xvars :
            if Xvars[i,j].X > 0.5 :
                solution.append((i,j))
                
        feasible, tour = LookForMinSubTour(solution,nodes)

        if feasible :
            stop = True 
        else:
            mod.addConstr(gp.quicksum(Xvars[i,j] for i in tour for j in tour if i != j) <= len(tour) -1)
            
    return solution,distance
    
def LookForSubTours(solution: List[Tuple[int,int]], firstNode: int,nodes: Dict[int, Tuple[float, float]]) -> Tuple[bool,List[int]]:

    feasible = True

    unVisited = list(nodes.keys())
    visited = []
    nextNode = firstNode
    
    while nextNode not in visited :
        
        currentNode = nextNode
        unVisited.remove(currentNode)
        visited.append(currentNode)
        
        for (i,j) in solution :
            if i == currentNode :
                nextNode = j
                break
        
    if len(unVisited) > 0 :
        feasible = False
        
    return feasible, visited

def LookForMinSubTour(solution: List[Tuple[int,int]],nodes: Dict[int, Tuple[float, float]]) -> Tuple[bool,List[tuple[int,int]]]:
    
    unVisited = list(nodes.keys())
    minTour = list(nodes.keys())
    
    while len(unVisited) > 0 :
        
        firstNode = unVisited[0]
        feasible, SubTour = LookForSubTours(solution, firstNode,nodes)
        
        if len(SubTour) <= len(minTour):
            minTour = SubTour
        
        for i in SubTour :
            unVisited.remove(i)
        
    return feasible, minTour

if __name__=="__main__":
    main()