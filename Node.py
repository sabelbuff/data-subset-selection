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
            sigma = self.dismatrix[self.neighbors[0].i_index, self.neighbors[0].j_index]
            self.out_msgs[0] = (self.damp * self.prev_out_msgs[0]) + ((1 - self.damp) * -sigma)

        elif self.nodetype == 'IC':
            i = self.neighbors[0].i_index
            eta = [self.neighbors[k].in_msgs[2] - self.dismatrix[i, k] for k in range(len(self.neighbors))]
            for j in range(len(self.neighbors)):
                temp_eta = eta[:]
                del temp_eta[j]
                self.out_msgs[j] = (self.damp * self.prev_out_msgs[j]) + ((1 - self.damp) * -max(temp_eta))

        else:
            j = self.neighbors[0].j_index
            dis_eta = [self.neighbors[k].in_msgs[1] - self.dismatrix[k, j] for k in range(len(self.neighbors))]
            dis_eta = np.array(dis_eta)
            for i in range(len(self.neighbors)):
                temp_dis_eta = dis_eta[:]
                np.delete(temp_dis_eta, i, 0)
                sum_value = np.sum(np.maximum(temp_dis_eta, 0))
                alpha = np.minimum(0, (-self.reg + sum_value))
                self.out_msgs[i] = (self.damp * self.prev_out_msgs[i]) + ((1 - self.damp) * alpha)
