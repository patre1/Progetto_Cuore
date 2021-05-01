from Strutture import *
from Nodi import *
import math
import random
from abc import abstractmethod, ABC
import copy


class BFS(ABC, object):

    def __init__(self, initial_state=None, goal_state=None, province=None):
        self.initial_state = initial_state
        self.goal_state = goal_state
        self.province=province
        self.queue = Queue()
        self.queue.addQueue(Node(initial_state, parent=None))
        self.path = []
        self.stop = False

    def run(self):
        while (True):
            if self.stop:
                return
            element = self.queue.deQueue()
            if (not self.checkGoal(element)):
                adjacents = self.getAdjacents(element)
                for adjacent in adjacents:
                    self.queue.addQueue(Node(adjacent, element))
            else:
                self.returnAnswer(element)
                return self.path

    @abstractmethod
    def getAdjacents(self, node):
        pass

    def checkGoal(self, node):
        if node.name == self.goal_state:
            return True
        else:
            return False

    def returnAnswer(self, node):
        while (node.parent != None):
            self.path.append(node.name)
            node = node.parent
        self.path.append(self.initial_state)
        self.path.reverse()


class DFS(ABC):

    def __init__(self, initial_state=None, goal_state=None,province=None):
        self.initial_state = initial_state
        self.goal_state = goal_state
        self.province=province
        self.stack = Stack()
        self.stack.push(Node(self.initial_state, parent=None))
        self.path = []
        self.stop = False

    def run(self):
        while (True):
            if self.stop:
                return
            element = self.stack.pop()
            if (not self.checkGoal(element)):
                adjacents = self.getAdjacents(element)
                random.shuffle(adjacents)
                for adjacent in adjacents:
                    self.stack.push(Node(adjacent, element))
            else:
                self.returnAnswer(element)
                return self.path

    @abstractmethod
    def getAdjacents(self, node):
        pass

    def checkGoal(self, node):
        if node.name == self.goal_state:
            return True
        else:
            return False

    def returnAnswer(self, node):
        while (node.parent != None):
            self.path.append(node.name)
            node = node.parent
        self.path.append(self.initial_state)
        self.path.reverse()


class UCS(ABC):

    def __init__(self, initial_state=None, goal_state=None,province=None):
        self.initial_state = initial_state
        self.goal_state = goal_state
        self.province=province
        self.priorityQueue = UCSMinheap()
        self.priorityQueue.insert(UCS_Node(initial_state, g_value=0, parent=None))
        self.path = []
        self.stop = False

    def run(self):
        while (True):
            if self.stop:
                return
            element = self.priorityQueue.delMin()
            if (not self.checkGoal(element)):
                adjacents = self.getAdjacents(element)
                for adjacent in adjacents:
                    adjacent_g_value = element.g_value + adjacent[1]
                    self.priorityQueue.insert(UCS_Node(adjacent[0], g_value=adjacent_g_value, parent=element))

            else:
                self.returnAnswer(element)
                return self.path

    @abstractmethod
    def getAdjacents(self, node):
        pass

    def checkGoal(self, node):
        if node.name == self.goal_state:
            return True
        else:
            return False

    def returnAnswer(self, node):
        while (node.parent != None):
            self.path.append(node.name)
            node = node.parent
        self.path.append(self.initial_state)
        self.path.reverse()

class DLS(ABC):

    def __init__(self, initial_state=None, goal_state=None,province=None,max_depth=0):
        self.initial_state = initial_state
        self.goal_state = goal_state
        self.province = province
        self.max_depth = max_depth
        self.stack = Stack()
        self.stack.push(DLS_Node(self.initial_state, depth=0, parent=None))
        self.path = []
        self.stop = False

    def run(self):
        while (True):
            if self.stop:
                return
            element = self.stack.pop()
            if element == None:
                return False
            if (not self.checkGoal(element)):
                if element.depth < self.max_depth:
                    adjacents = self.getAdjacents(element)
                    random.shuffle(adjacents)
                    for adjacent in adjacents:
                        self.stack.push(DLS_Node(adjacent, depth=element.depth + 1, parent=element))
            else:
                self.returnAnswer(element)
                return self.path

    @abstractmethod
    def getAdjacents(self, node):
        pass

    def checkGoal(self, node):
        if node.name == self.goal_state:
            return True
        else:
            return False

    def returnAnswer(self, node):
        while (node.parent != None):
            self.path.append(node.name)
            node = node.parent
        self.path.append(self.initial_state)
        self.path.reverse()


