from common import readNodes,calculateEdges,plotOrientedSolution
import gurobipy as gp
from gurobipy import GRB
from typing import List, Tuple,Dict,Any
import time

FILE_NAME = "lin318/lin318.tsp"

def main():

    Nodes = readNodes(FILE_NAME)

    dimension = len(Nodes)

    Dist = calculateEdges(Nodes,dimension)

    startTime = time.perf_counter()
    SOL,distance = solve(Nodes,Dist)
    endTime = time.perf_counter()  
      
    plotOrientedSolution(Nodes, SOL, title=f"Soluzione Ottima ATSP. Distance: {distance:.2f}. Time: {(endTime-startTime):.6f}s")

def solve(Nodes,Dist):
    mod = gp.Model("TSP")

    Xvars = mod.addVars(Dist.keys(), obj = Dist, vtype = GRB.BINARY, name = "x")

    outstar = mod.addConstrs(gp.quicksum(Xvars[i,j] for j in Nodes if i != j) == 1 for i in Nodes)

    instar = mod.addConstrs(Xvars.sum('*',i) == 1 for i in Nodes)
    
    mod.setParam("OutputFlag", 0)

    stop = False
    
    NumIt = 0

    distance = 0
    
    
    while not stop :
    
        NumIt = NumIt +1 

        mod.optimize()

        LowerBound = mod.ObjVal
        
        distance = mod.ObjVal

        SOL = []
        for (i,j) in Xvars :
            if Xvars[i,j].X > 0.5 :
                SOL.append((i,j))
                
        feasible, Tour = LookForMinSubTour(SOL,Nodes)

        if feasible :
            stop = True 
        else:
            mod.addConstr(gp.quicksum(Xvars[i,j] for i in Tour for j in Tour if i != j) <= len(Tour) -1)
        
        #print("Numero iterazione =", NumIt)
        #print("Lower Bound = ", LowerBound)
        #print("Soluzione corrente =", SOL)
        #print("Subtour individuato = ", Tour)
    return SOL,distance
    
def LookForSubTours(SOL, FirstNode,Nodes):

    feasible = True

    UnVisited = list(Nodes.keys())
    Visited = []
    NextNode = FirstNode
    
    while NextNode not in Visited :
        
        CurrentNode = NextNode
        UnVisited.remove(CurrentNode)
        Visited.append(CurrentNode)
        
        for (i,j) in SOL :
            if i == CurrentNode :
                NextNode = j
                break
        
    if len(UnVisited) > 0 :
        feasible = False
        
    return feasible, Visited

def LookForMinSubTour(SOL,Nodes):
    
    UnVisited = list(Nodes.keys())
    MinTour = list(Nodes.keys())
    
    while len(UnVisited) > 0 :
        
        FirstNode = UnVisited[0]
        feasible, SubTour = LookForSubTours(SOL, FirstNode,Nodes)
        
        if len(SubTour) <= len(MinTour):
            MinTour = SubTour
        
        for i in SubTour :
            UnVisited.remove(i)
        
    return feasible, MinTour

if __name__=="__main__":
    main()