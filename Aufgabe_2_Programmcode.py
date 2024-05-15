# -*- coding: utf-8 -*-
"""

"""

import pydot
import numpy as np
import matplotlib.pyplot as plt

class EVENT:
    def __init__(self,name,l):
        self.name = name
        self.fail = l
    # def add(self, node):
    #     return
    def failure_probability(self):
        return self.fail
        
class NOTNODE:
    def __init__(self,name):
        self.name = name
        self.nodes = []
        self.fail = 1
    def add(self,node):
        self.nodes.append(node)
        return
    def failure_probability(self):
        self.fail = 1
        for node in self.nodes:
            self.fail *= node.failure_probability()
        return 1 - self.fail
 
class ORNODE:
    def __init__(self,name):
        self.name = name
        self.nodes = []
        self.fail = 0
    def add(self,node):
        self.nodes.append(node)
        return
    def failure_probability(self):   
        self.fail = 0
        and_prob = 1
        for node in self.nodes:
            self.fail += node.failure_probability()
            and_prob *= node.failure_probability()    
        return self.fail - and_prob
      
class ANDNODE:
    def __init__(self,name):
        self.name = name
        self.nodes = []
        self.fail = 1
    def add(self,node):
        self.nodes.append(node)
        return
    def failure_probability(self):
        self.fail = 1
        for node in self.nodes:
            self.fail *= node.failure_probability()        
        return self.fail

# c)       
class GraphPrint:
   def __init__(self,name):     
        self.name = name
   def create(self,root):
       self.g = pydot.Dot(graph_type="digraph")
       def godeeperoradd(node, previousnode):
           if isinstance(node,EVENT):
              self.g.add_node(pydot.Node( node.name ) ) 
              self.g.add_edge(pydot.Edge(previousnode.name, node.name ) )
           elif isinstance(node, (ORNODE, ANDNODE,NOTNODE) ):
               self.g.add_node(pydot.Node( node.name  ) )
               self.g.add_edge(pydot.Edge(previousnode.name, node.name ) )
               if len(node.nodes) > 1:
                   for x in node.nodes: 
                       godeeperoradd(x, node)
               else: godeeperoradd(node.nodes[0], node)
               
       for node in root.nodes:
           godeeperoradd(node, root) 
                              
   def view(self):
       return self.g.write_png(self.name + "_Graph.png")

TOP = ANDNODE("TOP")
A = ORNODE("A")
  
E1 = EVENT("1", 0.1)    
E2 = EVENT("2", 0.1)    
E3 = EVENT("3", 0.1)    

TOP.add(A)
TOP.add(E1)
A.add(E2)
A.add(E3)    

g_test = GraphPrint("Test")
g_test.create(TOP)
g_test.view()   
        
# a)
A = EVENT("A", 0.01)    
B = EVENT("B", 0.1)    
C = EVENT("C", 0.001) 
D = EVENT("D", 0.01)    
E = EVENT("E", 0.01)    
F = EVENT("F", 0.01) 
G = EVENT("G", 0.1)    

K1 = ANDNODE("K1")    
K2 = ANDNODE("K2")   
K4 = ANDNODE("K4")   

K3 = ORNODE("K3")      
K5 = ORNODE("K5")    

K6 = NOTNODE("K6")  

K1.add(K2) 
K1.add(K6)   

K2.add(D)  
K2.add(E)  
K2.add(K4)     

K4.add(K5)    
K4.add(C) 

K5.add(A)  
K5.add(B) 

K6.add(K3)    

K3.add(F)  
K3.add(G)    

g_a = GraphPrint("A")
g_a.create(K1)
g_a.view()       
    

print ("Ausfallwahrscheinlichkeit an K1: " +  str( K1.failure_probability() ) )

# d)    
A_mean, A_sigma = 0.01, 0.002
B_mean, B_sigma = 0.1, 0.02
C_mean, C_sigma = 0.001, 0.002
D_mean, D_sigma = 0.01, 0.002
E_mean, E_sigma = 0.01, 0.002
F_mean, F_sigma = 0.01, 0.002
G_mean, G_sigma = 0.1, 0.02

n = 1000

failure_probabilities = np.zeros(n)
A_realisations = np.zeros(n)
B_realisations = np.zeros(n)
C_realisations = np.zeros(n)
D_realisations = np.zeros(n)
E_realisations = np.zeros(n)
F_realisations = np.zeros(n)
G_realisations = np.zeros(n)

for i in range(n):
    A_realisations[i] = max(0, np.random.normal(A_mean, A_sigma) )
    B_realisations[i] = max(0, np.random.normal(B_mean, B_sigma) )
    C_realisations[i] = max(0, np.random.normal(C_mean, C_sigma) )
    D_realisations[i] = max(0, np.random.normal(D_mean, D_sigma) )
    E_realisations[i] = max(0, np.random.normal(E_mean, E_sigma) )
    F_realisations[i] = max(0, np.random.normal(F_mean, F_sigma) )
    G_realisations[i] = max(0, np.random.normal(G_mean, G_sigma) )
    
    K5_probability =  A_realisations[i] + B_realisations[i] -  A_realisations[i] * B_realisations[i]
    K4_probability = K5_probability * C_realisations[i]
    K2_probability = D_realisations[i] * E_realisations[i] * K4_probability
    
    K3_probability =  F_realisations[i] +  G_realisations[i] - F_realisations[i] *  G_realisations[i]
    K6_probability = 1 - K3_probability
    
    K1_probability = K2_probability * K6_probability
    
    failure_probabilities[i] =  K1_probability

# Histogramm der Fehlerwahrscheinlichkeiten anzeigen
plt.hist(failure_probabilities, bins=30, edgecolor='black')
plt.xlabel('Fehlerwahrscheinlichkeit')
plt.ylabel('Anzahl der Versuche')
plt.title('Histogramm der Fehlerwahrscheinlichkeiten')
plt.grid(True)
plt.show()

print ("Ausfallwahrscheinlichkeit an K1: " +  str( failure_probabilities.mean() ) )
print ("Ausfallwahrscheinlichkeit A: " +  str( A_realisations.mean() ) )
print ("Ausfallwahrscheinlichkeit B: " +  str( B_realisations.mean() ) )
print ("Ausfallwahrscheinlichkeit C: " +  str( C_realisations.mean() ) )
print ("Ausfallwahrscheinlichkeit D: " +  str( D_realisations.mean() ) )
print ("Ausfallwahrscheinlichkeit E: " +  str( E_realisations.mean() ) )
print ("Ausfallwahrscheinlichkeit F: " +  str( F_realisations.mean() ) )
print ("Ausfallwahrscheinlichkeit G: " +  str( G_realisations.mean() ) )



