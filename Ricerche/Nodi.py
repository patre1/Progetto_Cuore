import math
import random
from abc import abstractmethod, ABC
import copy
import sys


class Node:

    def __init__(self, name, parent):
        self.name = name
        self.adjacents = []
        self.parent = parent

    def addAdjacent(self, node):
        self.adjacents.append(node)

    def getAdjacents(self):
        return self.adjacents


class DLS_Node:
    def __init__(self, name, depth, parent):
        self.name = name
        self.adjacents = []
        self.depth = depth
        self.parent = parent

    def addAdjacent(self, node):
        self.adjacents.append(node)

    def getAdjacents(self):
        return self.adjacents


class UCS_Node:

    def __init__(self, name, g_value, parent):
        self.name = name
        self.adjacents = {}
        self.g_value = g_value
        self.parent = parent

    def addAdjacent(self, node, distance):
        self.adjacents[node] = distance

    def getAdjacents(self):
        return self.adjacents



class Greedy_Node:

    def __init__(self, name, h_value, parent):
        self.name = name
        self.adjacents = []
        self.h_value = h_value
        self.parent = parent

    def addAdjacent(self, node):
        self.adjacents.append(node)

    def getAdjacents(self):
        return self.adjacents





