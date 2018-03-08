import numpy as np


class Node(object):

    epsilon = 10 ** (-4)
    VALID_NODES = {'X', 'IJ', 'IC', 'JF'}

    def __init__(self, node_type, nid, damp):
        self.nodetype = node_type
        self.nid = nid
        self.damp = damp
        self.neighbors = []
        self.in_msgs = []
        self.out_msgs = []
        self.prev_out_msgs = []

        self.validTypes(self.nodetype)

    @staticmethod
    def validTypes(arg):
        if arg not in Node.VALID_NODES:
            raise ValueError("The Node type must be one of : %r." % Node.VALID_NODES)

    def addNeighbors(self, node):
        self.neighbors.append(node)
        self.in_msgs.append(0)
        self.out_msgs.append(0)
        self.prev_out_msgs.append(0)
        node.neighbors.append(self)
        node.in_msgs.append(0)
        node.out_msgs.append(0)
        node.prev_out_msgs.append(0)
        # print("'fdjndf")

    def receiveMsg(self, sender, msg):
        index = self.neighbors.index(sender)
        self.in_msgs[index] = msg

    def sendMsg(self):
        for i in range(len(self.neighbors)):
            self.neighbors[i].receiveMsg(self, self.out_msgs[i])

    def checkConvergence(self):
        for i in range(len(self.out_msgs)):
            delta = np.absolute(self.out_msgs[i] - self.prev_out_msgs[i])
            if delta > Node.epsilon:
                return False

        return True


class Variable(Node):
    def __init__(self, node_type, nid, i, j, damp):
        super(Variable, self).__init__(node_type, nid, damp)
        self.i_index = i
        self.j_index = j

    def message(self):
        prev_out_msg = self.out_msgs[:]
        self.prev_out_msgs = prev_out_msg
        for i in range(len(self.in_msgs)):
            all_msgs = self.in_msgs[:]
            del all_msgs[i]
            self.out_msgs[i] = (self.damp * self.prev_out_msgs[i]) + ((1 - self.damp) * np.sum(all_msgs))


class Factor(Node):
    def __init__(self, nodetype, nid, varnodes, d, r, damp):
        super(Factor, self).__init__(nodetype, nid, damp)
        self.dismatrix = d
        self.reg = r
        self.N = len(d)

        for node in varnodes:
            self.addNeighbors(node)

    def message(self):
        prev_out_msg = self.out_msgs[:]
        self.prev_out_msgs = prev_out_msg
        if self.nodetype == 'IJ':
            for i in range(len(self.neighbors)):
                sigma = self.dismatrix[self.neighbors[i].i_index, self.neighbors[i].j_index]
                self.out_msgs[i] = (self.damp * self.prev_out_msgs[i]) + ((1 - self.damp) * -sigma)

        elif self.nodetype == 'IC':
            for j in range(len(self.neighbors)):
                # max_value = 0
                eta = []
                i = self.neighbors[j].i_index
                for k in range(self.N):
                    # index = 2
                    if not k == j:
                        # for n in range(len(self.neighbors[k].neighbors)):
                        #     if self.neighbors[k].neighbors[n].nodetype == 'FJ':
                        #         index = n
                        alpha = self.neighbors[k].in_msgs[2]
                        eta.append(alpha - self.dismatrix[i, k])
                        # if max_value < (alpha - self.dismatrix[i, k]):
                        #     max_value = alpha - self.dismatrix[i, k]
                # print(eta)
                # print(max(eta))
                self.out_msgs[j] = (self.damp * self.prev_out_msgs[j]) + ((1 - self.damp) * -max(eta))

        else:
            for i in range(len(self.neighbors)):
                sum_value = 0
                j = self.neighbors[i].j_index
                for k in range(self.N):
                    # index = 1
                    if not k == i:
                        # for n in range(len(self.neighbors[k].neighbors)):
                        #     if self.neighbors[k].neighbors[n].nodetype == 'IC':
                        #         index = n
                        eta = self.neighbors[k].in_msgs[1]
                        sum_value += max(0, eta - self.dismatrix[k, j])
                alpha = min(0, (-self.reg + sum_value))
                self.out_msgs[i] = (self.damp * self.prev_out_msgs[i]) + ((1 - self.damp) * alpha)


