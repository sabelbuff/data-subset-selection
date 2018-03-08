import numpy as np
from numpy import linalg as LA
from MessagePassingGraph import MessageGraph
from GreedyAlgorithm import Greedy
from ADMM import ADMM


class DS3(object):
    def __init__(self, dis_matrix, reg):
        self.reg = reg
        self.dis_matrix = dis_matrix

    def functionValue(self, z, p, a):
        temp_sum1 = 0
        if a == 0:
            for i in range(len(self.dis_matrix)):
                    temp_sum1 += max(z[i])
        else:
            for i in range(len(self.dis_matrix)):
                norm = LA.norm(z[i], ord=p)
                if norm > 0:
                    temp_sum1 += 1
        temp_sum2 = 0
        for i in range(len(self.dis_matrix)):
            for j in range(len(self.dis_matrix)):
                temp_sum2 += self.dis_matrix[i, j] * z[i, j]

        value = self.reg * temp_sum1 + temp_sum2

        return value

    def messagePassing(self, damp, max_iter):
        G = MessageGraph(self.dis_matrix, self.reg, damp)

        for i in range(len(self.dis_matrix)):
            var_node_i = []
            k = i
            for j in range(len(self.dis_matrix)):
                var_id = {'i': i, 'j': j}
                var_node = G.addVarNode(var_id, i, j)
                fac_id = {'i': i, 'j': j}
                G.addFacNode('IJ', fac_id, [var_node])
                var_node_i.append(var_node)
            fac_id = {'i': i, 'j': "Invalid"}
            G.addFacNode('IC', fac_id, var_node_i)

        for j in range(len(self.dis_matrix)):
            var_node_j = []
            for node in G.variables:
                if node.j_index == j:
                    var_node_j.append(node)
            fac_id = {'i': "Invalid", 'j': j}
            G.addFacNode('JF', fac_id, var_node_j)

        G.sumMax(max_iter)

        np.set_printoptions(threshold=np.nan)
        G.setVarValue()
        z_matrix = G.repmatrix

        function_value = self.functionValue(z_matrix.T, None, 0)

        data_rep = []
        count = 0
        for i in range(len(z_matrix)):
            flag = 0
            for j in range(len(z_matrix)):
                if z_matrix.T[i, j] == 1:
                    flag = 1
                    count += 1
            if flag == 1:
                data_rep.append(i)

        return data_rep, len(data_rep), function_value

    def greedyDeterministic(self):
        G = Greedy(self.dis_matrix, self.reg)
        rep_matrix, function_value = G.deterministic()
        data_rep = []
        for i in range(len(rep_matrix)):
            if rep_matrix[i] == 1:
                data_rep.append(i)

        return data_rep, len(data_rep), function_value

    def greedyRandomized(self):
        G = Greedy(self.dis_matrix, self.reg)
        rep_matrix, function_value = G.randomized()
        data_rep = []
        for i in range(len(rep_matrix)):
            if rep_matrix[i] == 1:
                data_rep.append(i)

        return data_rep, len(data_rep), function_value

    def ADMM(self, mu, epsilon, max_iter, p):
        G = ADMM(mu, epsilon, max_iter, self.reg)
        z_matrix = G.runADMM(self.dis_matrix, p)
        function_value = self.functionValue(z_matrix, p, 1)
        data_rep = []
        count = 0
        for i in range(len(z_matrix)):
            flag = 0
            for j in range(len(z_matrix)):
                if z_matrix[i, j] == 1:
                    flag = 1
                    count += 1
            if flag == 1:
                data_rep.append(i)

        return data_rep, len(data_rep), function_value

