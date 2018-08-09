from Node import Variable, Factor
import numpy as np


class MessageGraph(object):

    """
    :param d:    dis-similarity matrix for the dataset calculated based on euclideon distance.
    :param r:    regularization parameter.
    :param damp: message damp value to be used. This helps in faster convergence of the algorithm.
    """
    def __init__(self, d, r, damp):
        self.variables = []          # list to store all variables
        self.factors = []            # list to store all factors
        self.converged = False       # to check for convergence
        self.var_count = 0           # count of variables
        self.fac_count = 0           # count of variables
        self.dismatrix = d
        self.regvector = r
        self.damp = damp
        self.M = self.dismatrix.shape[0]
        self.N = self.dismatrix.shape[1]
        self.repmatrix = np.zeros((self.M, self.N))     # representative matrix




    """
        This function creates a variable node and adds it to the 'variables' list.

        :param nid: node id.
        :param i:   row index of the variable.
        :param j:   column index of the variable.

        :returns: variable node.
        """
    def addVarNode(self, nid, i, j):
        new_var = Variable('X', nid, i, j, self.damp)
        self.variables.append(new_var)
        self.var_count += 1

        return new_var




    """
        This function creates a factor node and adds it to the 'factors' list.
        
        :param nodetype: type of factor node { 'IJ', 'IC', 'JF'}
        :param nid:      node id.
        :param varnodes: list of variables nodes its connected to.   

        :returns: None.
        """
    def addFacNode(self, nodetype, nid, varnodes):
        new_fac = Factor(nodetype, nid, varnodes, self.dismatrix, self.regvector, self.damp)
        self.factors.append(new_fac)
        self.fac_count += 1




    """
        This function runs the sum-max message passing algorithm.

        :param iterations: number of iterations to run this algorithm.

        :returns: None.
        """
    def sumMax(self, iterations):
        temp = iterations
        while iterations > 0 and not self.converged:

            print("Iteration : ", (temp - iterations + 1))
            iterations -= 1

            # prepare and send the messages from factor nodes to variable nodes first and then vice-versa.
            for f in self.factors:
                f.message()
                f.sendMsg()

            for v in self.variables:
                v.message()
                v.sendMsg()

            # check for convergence after each iteration of message passing.
            flag = True

            if flag:
                for v in self.variables:
                    flag = v.checkConvergence()
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




    """
        This function calculates the belief of each variable by adding all te incoming messages from the
        factors its connected to.

        :param var: variable node.

        :returns: variable node.
        """
    def belief(self, var):
        belief = 0
        for i in range(len(var.in_msgs)):
            belief += var.in_msgs[i]

        return belief




    """
        This function sets the value of each element of representative matrix to either 0 or 1, based on the value of
        belief(1, if +ve, else 0) of the corresponding variable in the factor graph.

        :param: None.

        :returns: None.
        """
    def setVarValue(self):
        for v in self.variables:
            belief = self.belief(v)
            if belief > 0:
                self.repmatrix[v.i_index, v.j_index] = 1
            else:
                self.repmatrix[v.i_index, v.j_index] = 0