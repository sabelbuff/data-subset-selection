"""
This file contains the implementation of 'Dissimilarity-based Sparse Subset Selection' algorithm using different
types of optimization techniques such as, message passing, greedy algorithm, and ADMM.
"""

import numpy as np
from numpy import linalg as LA
from MessagePassingGraph import MessageGraph
from MessagePassingGraphSeq import MessageGraphSeq
from GreedyAlgorithm import Greedy
from ADMM import ADMM


np.set_printoptions(threshold=np.nan)

class DS3(object):
    """
    :param dis_matrix:  dis-similarity matrix for the dataset calculated based on euclideon distance.
    :param reg:         regularization parameter

    """
    def __init__(self, dis_matrix, reg):
        self.reg = reg
        self.dis_matrix = dis_matrix
        self.N = len(self.dis_matrix)

    def regCost(self, z, p):
        """
        This function calculates the total cost of choosing the as few representatives as possible.

        :param z: matrix whose non-zero rows corresponds to the representatives of the dataset.
        :param p: norm to be used to calculate regularization cost.

        :returns: regularization cost.
        """

        cost = 0
        for i in range(len(self.dis_matrix)):
            norm = LA.norm(z[i], ord=p)
            cost += norm

        return cost * self.reg

    def encodingCost(self, z):
        """
        This function calculates the total cost of encoding using all the representatives.

        :param z: matrix whose non-zero rows corresponds to the representatives of the dataset.

        :returns: encoding cost.
        """

        cost = 0
        for j in range(len(self.dis_matrix)):
            for i in range(len(self.dis_matrix)):
                cost += self.dis_matrix[i, j] * z[i, j]

        return cost

    def transitionCost(self, z, M, m0):
        """
        This function calculates the total cost of transitions between the representatives.

        :param z:  matrix whose non-zero rows corresponds to the representatives of the dataset.
        :param M:  transition probability matrix for the states in the source set.
        :param m0: initial probability vector of the states in the source set.

        :returns: transition cost.
        """

        sum1 = 0
        for i in range(1, self.N):
            sum1 += np.matmul(np.matmul(np.transpose(z[:,(i-1)]), M), z[:, i])
        sum2 = np.matmul(z[:, 1], m0)

        return sum1 + sum2

    def messagePassing(self, damp, max_iter):
        """
        This function finds the subset of data that can represent it as closely as possible given the
        regularization parameter. It uses message passing algorithm to solve the objective function for this problem,
        which is same as popular 'facility location problem'.

        To know more about this, please read :
        Solving the Uncapacitated Facility Location Problem Using Message Passing Algorithms
        by Nevena Lazic, Brendan J. Frey, Parham Aarabi
        http://proceedings.mlr.press/v9/lazic10a/lazic10a.pdf


        :param damp:      message damp value to be used. This helps in faster convergence of the algorithm.
        :param max_iter:  maximum number of iterations to run this algorithm.

        :returns: representative of the data, total number of representatives, and the objective function value.
        """

        M = self.dis_matrix.shape[0]
        N = self.dis_matrix.shape[1]

        # initiate a factor graph for the variables and factors.
        G = MessageGraph(self.dis_matrix, self.reg, damp)

        # add variable and factor nodes to the graph.
        for i in range(M):
            var_node_i = []

            for j in range(N):
                # add variable nodes for each variable.
                var_id = {'i': i, 'j': j}
                var_node = G.addVarNode(var_id, i, j)

                # add 'IJ' type factor node for each variable.
                fac_id = {'i': i, 'j': j}
                G.addFacNode('IJ', fac_id, [var_node])
                var_node_i.append(var_node)

            # add 'IC' type factor node for each row of variables.
            fac_id = {'i': i, 'j': "Invalid"}
            G.addFacNode('IC', fac_id, var_node_i)

        # add 'JF' type factor node for each column of variables.
        for j in range(N):
            var_node_j = []
            for node in G.variables:
                if node.j_index == j:
                    var_node_j.append(node)
            fac_id = {'i': "Invalid", 'j': j}
            G.addFacNode('JF', fac_id, var_node_j)

        print("Graph created")

        # apply sum-max message passing algorithm on the previously created graph.
        G.sumMax(max_iter)

        # set value of each variable in the representative matrix.
        G.setVarValue()
        z_matrix = G.repmatrix

        # calculate objective function value using the resulting representatives, given the regularization cost.
        obj_func_value = self.encodingCost(z_matrix) + self.regCost(z_matrix, np.inf)

        # find the index and total count of the representatives, given the representative matrix.
        data_rep = []
        count = 0
        print(z_matrix)
        for i in range(M):
            flag = 0
            for j in range(N):
                if z_matrix[i, j] == 1:
                    flag = 1
                    count += 1
            if flag == 1:
                data_rep.append(i)

        return data_rep, len(data_rep), obj_func_value

    def messagePassingSeq(self, damp, trans_matrix, init_prob_matrix, max_iter):
        """
        This function finds the subset of the data that can represent it as closely as possible given the
        regularization parameter and the underlying transition probabilities between the states. It uses
        message passing algorithm to solve the objective function for this problem, which is an extension for
        the popular 'facility location problem' and is called 'sequential facility location problem.

        To know more about this, please read :
        Subset Selection and Summarization in Sequential Data
        by Ehsan Elhamifar, M. Clara De Paolis Kaluza
        http://www.ccs.neu.edu/home/eelhami/publications/SeqSS-NIPS17-Ehsan.pdf

        :param damp:              message damp value to be used. This helps in faster convergence of the algorithm.
        :param trans_matrix:      transition probability matrix for the states in the source set.
        :param init_prob_matrix:  initial probability vector of the states in the source set.
        :param max_iter:          maximum number of iterations to run this algorithm.

        :returns: representative of the data, total number of representatives, and the objective function value.
        """

        M = self.dis_matrix.shape[0]
        T = self.dis_matrix.shape[1]

        # initiate a factor graph for the variables and factors.
        G = MessageGraphSeq(self.dis_matrix, self.reg, damp, trans_matrix, init_prob_matrix)

        # add variable and factor nodes to the graph.
        for i in range(M):
            var_node_i = []

            for t in range(T):
                # add variable nodes for each variable.
                var_id = {'i0': i, 't0': t, 'i1': "Invalid", 't1': "Invalid"}
                var_node = G.addVarNode(var_id, i, t)

                # add 'IT' type factor node for each variable.
                fac_id = {'i0': i, 't0': t, 'i1': "Invalid", 't1': "Invalid"}
                G.addFacNode('IT', fac_id, [var_node])
                var_node_i.append(var_node)

            # add 'IR' type factor node for each row of variables.
            fac_id = {'i0': i, 't0': "Invalid", 'i1': "Invalid", 't1': "Invalid"}
            G.addFacNode('IR', fac_id, var_node_i)

        # add 'CT' type factor node for each column of variables.
        for t in range(T):
            fac_id = {'i0': "Invalid", 't0': t, 'i1': "Invalid", 't1': "Invalid"}
            G.addFacNode('CT', fac_id, G.variables[:,t])

        # add 'D' type factor node for each pair of variable nodes in adjacent columns of variable matrix.
        for t in range(T-1):
            for i in range(M):
                for k in range(M):
                    fac_id = {'i0': i, 't0': t, 'i1': k, 't1': t+1}
                    G.addFacNode('D', fac_id, [G.variables[i, t], G.variables[k, t+1]])


        print("Graph created")

        # apply sum-max message passing algorithm on the previously created graph.
        G.sumMax(max_iter)

        # set value of each variable in the representative matrix.
        G.setVarValue()
        z_matrix = G.repmatrix

        # calculate objective function value using the resulting representatives, given the regularization cost
        # and the total cost of transitions between the states.
        obj_func_value = self.encodingCost(z_matrix) + self.regCost(z_matrix, np.inf) + \
                         self.transitionCost(z_matrix, trans_matrix, init_prob_matrix)

        # find the index and total count of the representatives, given the representative matrix.
        data_rep = []
        count = 0
        for i in range(M):
            flag = 0
            for t in range(T):
                if z_matrix.T[i, t] == 1:
                    flag = 1
                    count += 1
            if flag == 1:
                data_rep.append(i)

        return data_rep, len(data_rep), obj_func_value

    def greedyDeterministic(self):
        """
        This function finds the subset of the data that can represent it as closely as possible given the
        regularization parameter. It uses deterministic greedy algorithm on the sub-modular set of data to
        solve the objective which closely resembles the popular 'facility location problem'.

        To know more about this, please read :
        A TIGHT LINEAR TIME (1/2)-APPROXIMATION FOR UNCONSTRAINED SUBMODULAR MAXIMIZATION
        by NIV BUCHBINDER, MORAN FELDMAN, JOSEPH (SEFFI) NAOR, AND ROY SCHWARTZ
        https://www.openu.ac.il/personal_sites/moran-feldman/publications/SICOMP2015.pdf

        :param : None

        :returns: representative of the data, total number of representatives, and the objective function value.
        """

        # initialize the Greedy class.
        G = Greedy(self.dis_matrix, self.reg)

        # run the deterministic greedy algorithm.
        rep_matrix, obj_func_value = G.deterministic()

        return rep_matrix, len(rep_matrix), obj_func_value

    def greedyRandomized(self):
        """
        This function finds the subset of the data that can represent it as closely as possible given the
        regularization parameter. It uses randomized greedy algorithm on the sub-modular set of data to
        solve the objective which closely resembles the popular 'facility location problem'.

        To know more about this, please read :
        A TIGHT LINEAR TIME (1/2)-APPROXIMATION FOR UNCONSTRAINED SUBMODULAR MAXIMIZATION
        by NIV BUCHBINDER, MORAN FELDMAN, JOSEPH (SEFFI) NAOR, AND ROY SCHWARTZ
        https://www.openu.ac.il/personal_sites/moran-feldman/publications/SICOMP2015.pdf

        :param : None

        :returns: representative of the data, total number of representatives, and the objective function value.
        """

        # initialize the Greedy class.
        G = Greedy(self.dis_matrix, self.reg)

        # run the randomized greedy algorithm.
        rep_matrix, obj_func_value = G.randomized()

        return rep_matrix, len(rep_matrix), obj_func_value

    def ADMM(self, mu, epsilon, max_iter, p):
        """
        This function finds the subset of the data that can represent it as closely as possible given the
        regularization parameter. It uses 'alternating direction methods of multipliers' (ADMM) algorithm to
        solve the objective function for this problem, which is similar to the popular 'facility location problem'.

        To know more about this, please read :
        Dissimilarity-based Sparse Subset Selection
        by Ehsan Elhamifar, Guillermo Sapiro, and S. Shankar Sastry
        https://arxiv.org/pdf/1407.6810.pdf

        :param mu:        penalty parameter.
        :param epsilon:   small value to check for convergence.
        :param max_iter:  maximum number of iterations to run this algorithm.
        :param p:         norm to be used.

        :returns: representative of the data, total number of representatives, and the objective function value.
        """

        # initialize the ADMM class.
        G = ADMM(mu, epsilon, max_iter, self.reg)

        # run the ADMM algorithm.
        z_matrix = G.runADMM(self.dis_matrix, p)

        # new representative matrix obtained after changing largest value in each column to 1 and other values to 0.
        new_z_matrix = np.zeros(np.array(z_matrix).shape)
        idx = np.argmax(z_matrix, axis=0)
        for k in range(len(z_matrix)):
            new_z_matrix[idx[k], k] = 1

        # obj_func_value = self.encodingCost(z_matrix) + self.regCost(z_matrix, p)
        obj_func_value = self.encodingCost(z_matrix) + self.regCost(z_matrix, np.inf)

        # obj_func_value_post_proc = self.encodingCost(new_z_matrix) + self.regCost(new_z_matrix, p)
        obj_func_value_post_proc = self.encodingCost(new_z_matrix) + self.regCost(new_z_matrix, np.inf)

        # find the index and total count of the representatives, given the representative matrix.
        data_rep = []
        count = 0
        for i in range(len(z_matrix)):
            flag = 0
            for j in range(len(z_matrix)):
                if z_matrix[i, j] > 0.1:
                    flag = 1
                    count += 1
            if flag == 1:
                data_rep.append(i)

        return data_rep, len(data_rep), obj_func_value, obj_func_value_post_proc

