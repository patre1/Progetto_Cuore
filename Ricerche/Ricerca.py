from graph import *
from Algoritmi import *
import math
import random


class RicercaBFS(BFS):

    def __init__(self, initial_state='', goal_state='', province=''):
        if(province =="TA"):
            self.graph = grafo.provinciaTA
        elif(province=="BA"):
            self.graph=grafo.provinciaBA
        elif (province == "LE"):
            self.graph = grafo.provinciaLE
        elif (province == "BT"):
            self.graph = grafo.provinciaBT
        elif (province == "BR"):
            self.graph = grafo.provinciaBR
        elif (province == "FG"):
            self.graph = grafo.provinciaFG
        super().__init__(initial_state=initial_state, goal_state=goal_state,province=self.graph)

    def getAdjacents(self, node):
        temp = []
        adjacents = self.graph[node.name]
        for pair in adjacents:
            temp.append(list(pair.keys())[0])

        return temp


class RicercaDFS(DFS):

    def __init__(self, initial_state='', goal_state='',province=''):
        if (province == "TA"):
            self.graph = grafo.provinciaTA
        elif (province == "BA"):
            self.graph = grafo.provinciaBA
        elif (province == "LE"):
            self.graph = grafo.provinciaLE
        elif (province == "BT"):
            self.graph = grafo.provinciaBT
        elif (province == "BR"):
            self.graph = grafo.ProvinciaBR
        elif (province == "FG"):
            self.graph = grafo.provinciaFG
        super().__init__(initial_state=initial_state, goal_state=goal_state,province=self.graph)

    def getAdjacents(self, node):
        temp = []
        adjacents = self.graph[node.name]
        for pair in adjacents:
            temp.append(list(pair.keys())[0])

        return temp


class RicercaUCS(UCS):

    def __init__(self, initial_state='', goal_state='',province=''):
        if (province == "TA"):
            self.graph = grafo.provinciaTA
        elif (province == "BA"):
            self.graph = grafo.provinciaBA
        elif (province == "LE"):
            self.graph = grafo.provinciaLE
        elif (province == "BT"):
            self.graph = grafo.provinciaBT
        elif (province == "BR"):
            self.graph = grafo.provinciaBR
        elif (province == "FG"):
            self.graph = grafo.provinciaFG
        super().__init__(initial_state=initial_state, goal_state=goal_state,province=self.graph)

    def getAdjacents(self, node):
        temp = []
        adjacents = self.graph[node.name]
        for pair in adjacents:
            name = list(pair.keys())[0]
            g_value = list(pair.values())[0]
            temp.append([name, g_value])

        return temp

class RicercaDLS(DLS):

    def __init__(self, initial_state='', goal_state='', max_depth=0,province=''):
        if (province == "TA"):
            self.graph = grafo.provinciaTA
        elif (province == "BA"):
            self.graph = grafo.provinciaBA
        elif (province == "LE"):
            self.graph = grafo.provinciaLE
        elif (province == "BT"):
            self.graph = grafo.provinciaBT
        elif (province == "BR"):
            self.graph = grafo.provinciaBR
        elif (province == "FG"):
            self.graph = grafo.provinciaFG
        super().__init__(initial_state=initial_state, goal_state=goal_state, province=self.graph, max_depth=max_depth)

    def getAdjacents(self, node):
        temp = []
        adjacents = self.graph[node.name]
        for pair in adjacents:
            temp.append(list(pair.keys())[0])

        return temp


class RicercaIDS:

    def __init__(self, initial_state='', goal_state='',):
        self.initial_state = initial_state
        self.goal_state = goal_state
        self.depth = 0

        self.dls = None

    def run(self,province=''):
        while (True):
            self.dls = RicercaDLS(max_depth=self.depth,province='')
            found_answer = self.dls.run()
            if found_answer:
                return found_answer
            else:
                self.depth += 1

    def makeGraph(self):
        graph = {}
        for i in range(9):
            for j in range(9):
                graph['A_' + str(i) + str(j)] = []

        for i in range(9):
            for j in range(9):
                index = 'A_' + str(i) + str(j)
                subtable_row = i // 3
                subtable_col = j // 3

                for row in range(9):
                    if i != row:
                        graph[index].append('A_' + str(row) + str(j))

                for col in range(9):
                    if j != col:
                        graph[index].append('A_' + str(i) + str(col))

                subtable_row = subtable_row * 3
                subtable_col = subtable_col * 3
                for k in range(subtable_row, subtable_row + 3):
                    for t in range(subtable_col, subtable_col + 3):
                        if i != k or j != t:
                            if 'A_' + str(k) + str(t) not in graph[index]:
                                graph[index].append('A_' + str(k) + str(t))

        return graph



