import numpy as np

# generic Node class
class Node(object):

    epsilon = 10 ** (-4)                        # small value to check for the convergence
    VALID_NODES = {'X', 'IJ', 'IC', 'JF'}       # set of Node types

    """
        :param node_type:  type of the Node.
        :param nid:        Node id.
        :param damp:       message damp value to be used. This helps in faster convergence of the algorithm.
        """
    def __init__(self, node_type, nid, damp):
        self.nodetype = node_type
        self.nid = nid
        self.damp = damp
        self.neighbors = []                 # list of connected nodes to this node
        self.in_msgs = []                   # list of incoming messages
        self.out_msgs = []                  # list of outgoing messages
        self.prev_out_msgs = []             # list to maintain previous outgoing messages

        self.validTypes(self.nodetype)




    """
        This function checks if the Node type is valid one or not.
        
        :param arg: variable to represent a valid node type.
        
        :raise: Invalid Node.
        """
    @staticmethod
    def validTypes(arg):
        if arg not in Node.VALID_NODES:
            raise ValueError("The Node type must be one of : %r." % Node.VALID_NODES)




    """
        This function adds the connected node to the neighbors list and adds itself to the node's neighbors list.
        It also initialize the incoming and outgoing messages to 0 for each connected node.
        
        :param node: a variable Node.
        """
    def addNeighbors(self, node):
        self.neighbors.append(node)
        self.in_msgs.append(0)
        self.out_msgs.append(0)
        self.prev_out_msgs.append(0)
        node.neighbors.append(self)
        node.in_msgs.append(0)
        node.out_msgs.append(0)
        node.prev_out_msgs.append(0)




    """
        This function receives the message from te sender node and updates te incoming message index corresponding
        to that node.

        :param sender: message sender node.
        :param msg:    message to be updated.
        """
    def receiveMsg(self, sender, msg):
        index = self.neighbors.index(sender)
        self.in_msgs[index] = msg




    """
        This function sends message contained in its outgoing message list to each of its corresponding neigbors.

        :param : None
        """
    def sendMsg(self):
        for i in range(len(self.neighbors)):
            self.neighbors[i].receiveMsg(self, self.out_msgs[i])




    """
        This function checks the convergence by comparing the difference of the current outgoinf message to the
        previous outgoing message.

        :param : None
        
        :returns: True, if all outgoing messages are within the epsilon diff to previous messages.
        """
    def checkConvergence(self):
        for i in range(len(self.out_msgs)):
            delta = np.absolute(self.out_msgs[i] - self.prev_out_msgs[i])
            if delta > Node.epsilon:
                return False

        return True



# Variable Node class
class Variable(Node):

    """
    :param nodetype:    type of node.
    :param nid:         Node Id.
    :param i:           row index of the variable node.
    :param j:           column index of the variable node.
    :param damp:        message damp value to be used. This helps in faster convergence of the algorithm.
    """
    def __init__(self, node_type, nid, i, j, damp):
        super(Variable, self).__init__(node_type, nid, damp)
        self.i_index = i
        self.j_index = j




    """
        This function prepares the message from a variable node to all its connected factor nodes and saves to
        the outgoing message list. It also damps the new message update by the given damping factor and adds it to the 
        previous message to get the new outgoing message.

        :param : None
        """
    def message(self):
        prev_out_msg = self.out_msgs[:]
        self.prev_out_msgs = prev_out_msg
        for i in range(len(self.in_msgs)):
            all_msgs = self.in_msgs[:]
            del all_msgs[i]
            self.out_msgs[i] = (self.damp * self.prev_out_msgs[i]) + ((1 - self.damp) * np.sum(all_msgs))




# Factor Node class
class Factor(Node):

    """
    :param nodetype:    type of the node.
    :param nid:         Node id.
    :param varnodes:    list of variable nodes connected thhis factor node.
    :param d:           dis-similarity matrix.
    :param r:           regularization parameter.
    :param damp:        message damp value to be used. This helps in faster convergence of the algorithm.
    """
    def __init__(self, nodetype, nid, varnodes, d, r, damp):
        super(Factor, self).__init__(nodetype, nid, damp)
        self.dismatrix = d
        self.reg = r
        self.M = d.shape[0]
        self.N = d.shape[1]

        # add each node present in varnodes to the neighbors list.
        for node in varnodes:
            self.addNeighbors(node)




    """
            This function prepares the message from a factor node to all its connected variable nodes and saves to
            the outgoing message list. It also damps the new message update by the given damping factor and 
            adds it to the previous message to get the new outgoing message.

            :param : None
            """
    def message(self):
        prev_out_msg = self.out_msgs[:]
        self.prev_out_msgs = prev_out_msg

        # updates message for the 'IJ' type factor node.
        if self.nodetype == 'IJ':
            sigma = self.dismatrix[self.neighbors[0].i_index, self.neighbors[0].j_index]
            self.out_msgs[0] = (self.damp * self.prev_out_msgs[0]) + ((1 - self.damp) * -sigma)

        # updates message for the 'IC' type factor node.
        # updates message from a factor node to all variable nodes in a row connected to it.
        elif self.nodetype == 'IC':
            i = self.neighbors[0].i_index

            # take sum of all messages coming to this node.
            eta = [self.neighbors[k].in_msgs[2] - self.dismatrix[i, k] for k in range(len(self.neighbors))]

            # to calculate eta, use sum of all messages coming to this node, but not from the node to which outgoing
            # message is to be calculated.
            for j in range(len(self.neighbors)):
                temp_eta = eta[:]
                del temp_eta[j]
                self.out_msgs[j] = (self.damp * self.prev_out_msgs[j]) + ((1 - self.damp) * -max(temp_eta))

        # updates message for the 'JF' type factor node.
        # updates message from a factor node to all variable nodes in a column connected to it.
        else:
            j = self.neighbors[0].j_index

            # take sum of all messages coming to this node.
            dis_eta = [self.neighbors[k].in_msgs[1] - self.dismatrix[k, j] for k in range(len(self.neighbors))]
            dis_eta = np.array(dis_eta)

            # to calculate alpha, use sum of all messages coming to this node, but not from the node to which outgoing
            # message is to be calculated.
            for i in range(len(self.neighbors)):
                temp_dis_eta = dis_eta[:]
                np.delete(temp_dis_eta, i, 0)
                sum_value = np.sum(np.maximum(temp_dis_eta, 0))
                alpha = np.minimum(0, (-self.reg + sum_value))
                self.out_msgs[i] = (self.damp * self.prev_out_msgs[i]) + ((1 - self.damp) * alpha)
