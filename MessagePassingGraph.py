from Node import Variable, Factor
import numpy as np
import time


class MessageGraph(object):
    def __init__(self, d, r, damp):
        self.variables = []
        self.factors = []
        self.converged = False
        self.var_count = 0
        self.fac_count = 0
        self.dismatrix = d
        self.regvector = r
        self.damp = damp
        self.N = len(self.dismatrix)
        self.repmatrix = np.zeros((self.N, self.N))

    def addVarNode(self, nid, i, j):
        new_var = Variable('X', nid, i, j, self.damp)
        self.variables.append(new_var)
        self.var_count += 1

        return new_var

    def addFacNode(self, nodetype, nid, varnodes):
        new_fac = Factor(nodetype, nid, varnodes, self.dismatrix, self.regvector, self.damp)
        self.factors.append(new_fac)
        self.fac_count += 1

    def sumMax(self, iterations):
        temp = iterations
        while iterations > 0 and not self.converged:

            print("Iteration : ", (temp - iterations + 1))
            iterations -= 1
            for f in self.factors:
                f.message()
                f.sendMsg()

            for v in self.variables:
                v.message()
                v.sendMsg()

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

    def belief(self, var):
        belief = 0
        for i in range(len(var.in_msgs)):
            belief += var.in_msgs[i]

        return belief

    def setVarValue(self):
        for v in self.variables:
            belief = self.belief(v)
            if belief > 0:
                self.repmatrix[v.i_index, v.j_index] = 1
            else:
                self.repmatrix[v.i_index, v.j_index] = 0