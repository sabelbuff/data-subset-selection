from NodeSeq import Variable, Factor
import numpy as np
import time


class MessageGraphSeq(object):
    """
    :param d:    dis-similarity matrix for the dataset calculated based on euclideon distance.
    :param r:    regularization parameter.
    :param damp: message damp value to be used. This helps in faster convergence of the algorithm.
    :param M:    transition probability matrix for the states in the source set.
    :param m:   initial probability vector of the states in the source set.
    """
    def __init__(self, d, r, damp, M, m):
        self.factors = []                   # list to store all factors
        self.converged = False              # to check for convergence
        self.var_count = 0                  # count of variables
        self.fac_count = 0                  # count of variables
        self.dismatrix = d
        self.regvector = r
        self.trans_matrix = M
        self.init_prob_matrix = m
        self.damp = damp
        self.M = d.shape[0]
        self.T = d.shape[1]
        self.variables = np.empty((self.M, self.T), dtype=object)   # matrix to store all variables
        self.repmatrix = np.zeros((self.M, self.T))                 # representative matrix

    def addVarNode(self, nid, i, t):
        """
        This function creates a variable node and adds it to the 'variables' list.

        :param nid: node id.
        :param i:   row index of the variable.
        :param j:   column index of the variable.

        :returns: variable node.
        """

        new_var = Variable('X', nid, i, t, self.damp)
        self.variables[i, t] = new_var
        self.var_count += 1

        return new_var

    def addFacNode(self, nodetype, nid, varnodes):
        """
        This function creates a factor node and adds it to the 'factors' list.

        :param nodetype: type of factor node { 'IJ', 'IC', 'JF'}
        :param nid:      node id.
        :param varnodes: list of variables nodes its connected to.

        :returns: None.
        """

        new_fac = Factor(nodetype, nid, varnodes, self.dismatrix, self.trans_matrix, self.init_prob_matrix,
                         self.regvector, self.damp)
        self.factors.append(new_fac)
        self.fac_count += 1

    def sumMax(self, iterations):
        """
        This function runs the sum-max message passing algorithm.

        :param iterations: number of iterations to run this algorithm.

        :returns:
        """

        temp = iterations
        while iterations > 0 and not self.converged:

            print("Iteration : ", (temp - iterations + 1))
            iterations -= 1

            # prepare and send the messages from factor nodes to variable nodes first and then vice-versa.
            for f in self.factors:
                f.message()
                f.sendMsg()

            for v in self.variables:
                for v1 in v:
                    v1.message()
                    v1.sendMsg()

            # check for convergence after each iteration of message passing.
            flag = True

            if flag:
                for v in self.variables:
                    for v1 in v:
                        flag = v1.checkConvergence()
                        if not flag:
                            break

            if flag:
                for f in self.factors:
                    flag = f.checkConvergence()
                    if not flag:
                        break

            if flag:
                self.converged = True

        if not self.converged:
            print("sum-max algorithm did not converge, try running for more number of iterations.")

    def belief(self, var):
        """
        This function calculates the belief of each variable by adding all te incoming messages from the
        factors its connected to.

        :param var: variable node.

        :returns: variable node.
        """

        belief = 0
        for i in range(len(var.in_msgs)):
            belief += var.in_msgs[i]

        return belief

    def setVarValue(self):
        """
        This function sets the value of each element of representative matrix to either 0 or 1, based on the value of
        belief(1, if +ve, else 0) of the corresponding variable in the factor graph.

        :param: None.

        :returns
        """

        for v in self.variables:
            for v1 in v:
                belief = self.belief(v1)
                if belief > 0:
                    self.repmatrix[v1.i_index, v1.t_index] = 1
                else:
                    self.repmatrix[v1.i_index, v1.t_index] = 0