import numpy as np

# generic Node class
class Node(object):
    """
    :param node_type:  type of the Node.
    :param nid:        Node id.
    :param damp:       message damp value to be used. This helps in faster convergence of the algorithm.
    """

    epsilon = 10 ** (-4)                                # small value to check for the convergence
    VALID_NODES = {'X', 'IT', 'CT', 'IR', 'D'}          # set of Node types

    def __init__(self, node_type, nid, damp):
        # in_msgs, out_msgs, prev_out_msgs lists follow order of sigma, alpha, eta, gamma', gamma.
        self.nodetype = node_type
        self.nid = nid
        self.damp = damp
        self.neighbors = []                         # list of connected nodes to this node
        self.in_msgs = []                           # list of incoming messages
        self.out_msgs = []                          # list of outgoing messages
        self.prev_out_msgs = []                     # list to maintain previous outgoing messages

        self.validTypes(self.nodetype)

    @staticmethod
    def validTypes(arg):
        """
        This function checks if the Node type is valid one or not.

        :param arg: variable to represent a valid node type.

        :raise: Invalid Node.
        """

        if arg not in Node.VALID_NODES:
            raise ValueError("The Node type must be one of : %r." % Node.VALID_NODES)

    def addNeighbors(self, node):
        """
        This function adds the connected node to the neighbors list and adds itself to the node's neighbors list.
        It also initialize the incoming and outgoing messages to 0 for each connected node.

        :param node: a variable Node.
        """

        self.neighbors.append(node)
        self.in_msgs.append(0)
        self.out_msgs.append(0)
        self.prev_out_msgs.append(0)
        node.neighbors.append(self)
        node.in_msgs.append(0)
        node.out_msgs.append(0)
        node.prev_out_msgs.append(0)

    def receiveMsg(self, sender, msg):
        """
        This function receives the message from te sender node and updates te incoming message index corresponding
        to that node.

        :param sender: message sender node.
        :param msg:    message to be updated.
        """

        index = self.neighbors.index(sender)
        self.in_msgs[index] = msg

    def sendMsg(self):
        """
        This function sends message contained in its outgoing message list to each of its corresponding neigbors.

        :param : None
        """

        for i in range(len(self.neighbors)):
            self.neighbors[i].receiveMsg(self, self.out_msgs[i])

    def checkConvergence(self):
        """
        This function checks the convergence by comparing the difference of the current outgoinf message to the
        previous outgoing message.

        :param : None

        :returns: True, if all outgoing messages are within the epsilon diff to previous messages.
        """

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
    :param t:           column index of the variable node.
    :param damp:        message damp value to be used. This helps in faster convergence of the algorithm.
    """

    def __init__(self, node_type, nid, i, t, damp):
        super(Variable, self).__init__(node_type, nid, damp)
        self.i_index = i
        self.t_index = t

    def message(self):
        """
        This function prepares the message from a variable node to all its connected factor nodes and saves to
        the outgoing message list. It also damps the new message update by the given damping factor and adds it to the
        previous message to get the new outgoing message.

        :param : None
        """

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
    :param M:           transition probability matrix for the states in the source set.
    :param m:           initial probability vector of the states in the source set.
    """

    def __init__(self, nodetype, nid, varnodes, d, M, m, r, damp):
        super(Factor, self).__init__(nodetype, nid, damp)
        self.dis_matrix = d
        self.trans_matrix = M
        self.init_prob_matrix = m
        self.reg = r
        self.M = d.shape[0]
        self.T = d.shape[1]
        self.dis_matrix_star = self.dis_matrix
        self.dis_matrix_star[:, 0] = self.dis_matrix_star[:, 0] - self.init_prob_matrix

        # add each node present in varnodes to the neighbors list.
        for node in varnodes:
            self.addNeighbors(node)

    def message(self):
        """
        This function prepares the message from a factor node to all its connected variable nodes and saves to
        the outgoing message list. It also damps the new message update by the given damping factor and
        adds it to the previous message to get the new outgoing message.

        To know how sigma, alpha, eta, gamma and gamma' updates work, please read:
        Subset Selection and Summarization in Sequential Data
        by Ehsan Elhamifar, M. Clara De Paolis Kaluza
        http://www.ccs.neu.edu/home/eelhami/publications/SeqSS-NIPS17-Ehsan.pdf

        :param : None
        """

        prev_out_msg = self.out_msgs[:]
        self.prev_out_msgs = prev_out_msg

        # updates message for the 'IT' type factor node.
        if self.nodetype == 'IT':
            sigma = self.dis_matrix_star[self.neighbors[0].i_index, self.neighbors[0].t_index]
            self.out_msgs[0] = (self.damp * self.prev_out_msgs[0]) + ((1 - self.damp) * -sigma)

        # updates message for the 'CT' type factor node.
        # updates message from a factor node to all variable nodes in a row connected to it.
        elif self.nodetype == 'CT':
            t = self.neighbors[0].t_index
            eta = np.zeros(self.M)

            # variables in the columns on the edges do not receive either gamma or gamma' messages.
            if t == 0 or t == self.T - 1:
                for k in range(self.M):
                    one_gamma_sum = np.sum(self.neighbors[k].in_msgs[3:])
                    eta[k] = self.neighbors[k].in_msgs[1] - self.dis_matrix_star[k, t] + one_gamma_sum

            # variables in the inner columns receive both gamma and gamma' messages.
            else:
                for k in range(self.M):
                    gamma_prime_sum = np.sum(self.neighbors[k].in_msgs[3:3+self.M])
                    gamma_sum = np.sum(self.neighbors[k].in_msgs[3+self.M:])
                    eta[k] = self.neighbors[k].in_msgs[1] - self.dis_matrix_star[k, t] + gamma_sum + gamma_prime_sum

            # to calculate eta, use sum of all messages coming to this node, but not from the node to which outgoing
            # message is to be calculated.
            for i in range(self.M):
                temp_eta = eta[:]
                np.delete(temp_eta, i, 0)
                self.out_msgs[i] = (self.damp * self.prev_out_msgs[i]) + ((1 - self.damp) * -max(temp_eta))

        # updates message for the 'IR' type factor node.
        # updates message from a factor node to all variable nodes in a column connected to it.
        elif self.nodetype == 'IR':
            i = self.neighbors[0].i_index
            dis_eta = np.zeros(self.T)

            for k in range(self.T):
                # variables at the ends of the row do not receive gamma or gamma' messages.
                if k == 0 or k == self.T - 1:
                    one_gamma_sum = np.sum(self.neighbors[k].in_msgs[3:])
                    dis_eta[k] = self.neighbors[k].in_msgs[2] - self.dis_matrix_star[i, k] + one_gamma_sum

                # variables in the inner columns of the row receive both gamma and gamma' messages.
                else:
                    gamma_prime_sum = np.sum(self.neighbors[k].in_msgs[3:3+self.M])
                    gamma_sum = np.sum(self.neighbors[k].in_msgs[3+self.M:])
                    dis_eta[k] = self.neighbors[k].in_msgs[2] - self.dis_matrix_star[i, k] + gamma_sum + gamma_prime_sum

            dis_eta = np.array(dis_eta)

            # to calculate eta, use sum of all messages coming to this node, but not from the node to which outgoing
            # message is to be calculated.
            for t in range(self.T):
                temp_dis_eta = dis_eta[:]
                np.delete(temp_dis_eta, t, 0)
                sum_value = np.sum(np.maximum(temp_dis_eta, 0))
                alpha = np.minimum(0, (-self.reg + sum_value))
                self.out_msgs[t] = (self.damp * self.prev_out_msgs[t]) + ((1 - self.damp) * alpha)

        # updates message for the 'D' type factor node.
        # updates message from a factor node to all variable nodes in a column connected to it.
        # calculate rho and rho' for the gamma and gamma' messages respectively.
        else:
            i_0 = self.neighbors[0].i_index
            t_0 = self.neighbors[0].t_index
            i_1 = self.neighbors[1].i_index
            t_1 = self.neighbors[1].t_index

            # variables present in the first column do not receive gamma' messages.
            if t_0 == 0:
                temp_gammas_prime = self.neighbors[1].in_msgs[3:3+self.M]
                temp_gammas = self.neighbors[0].in_msgs[3:]
                del temp_gammas_prime[i_0]
                del temp_gammas[i_1]
                rho = -self.dis_matrix_star[i_1, t_1] + self.neighbors[1].in_msgs[1] + self.neighbors[1].in_msgs[2] + \
                      np.sum(self.neighbors[1].in_msgs[3+self.M:]) + np.sum(temp_gammas_prime)
                rho_prime = -self.dis_matrix_star[i_0, t_0] + self.neighbors[0].in_msgs[1] + \
                            self.neighbors[0].in_msgs[2] + np.sum(temp_gammas)

            # variables present in the last column do not receive gamma messages.
            elif t_1 == (self.T - 1):
                temp_gammas_prime = self.neighbors[1].in_msgs[3:]
                temp_gammas = self.neighbors[0].in_msgs[3 + self.M:]
                del temp_gammas_prime[i_0]
                del temp_gammas[i_1]
                rho = -self.dis_matrix_star[i_1, t_1] + self.neighbors[1].in_msgs[1] + self.neighbors[1].in_msgs[2] + \
                        np.sum(temp_gammas_prime)
                rho_prime = -self.dis_matrix_star[i_0, t_0] + self.neighbors[0].in_msgs[1] + \
                            self.neighbors[0].in_msgs[2] + np.sum(temp_gammas) + \
                            np.sum(self.neighbors[0].in_msgs[3:3 + self.M])

            # variables present in the inbetween column receive both gamma and gamma' messages.
            else:
                temp_gammas_prime = self.neighbors[1].in_msgs[3:3 + self.M]
                temp_gammas = self.neighbors[0].in_msgs[3 + self.M:]
                del temp_gammas_prime[i_0]
                del temp_gammas[i_1]
                rho = -self.dis_matrix_star[i_1, t_1] + self.neighbors[1].in_msgs[1] + self.neighbors[1].in_msgs[2] + \
                      np.sum(self.neighbors[1].in_msgs[3 + self.M:]) + np.sum(temp_gammas_prime)
                rho_prime = -self.dis_matrix_star[i_0, t_0] + self.neighbors[0].in_msgs[1] + \
                            self.neighbors[0].in_msgs[2] + np.sum(temp_gammas) + \
                            np.sum(self.neighbors[0].in_msgs[3:3 + self.M])

            # update gamma and gamma' message using the previously calculated rho and rho'.
            gamma = np.maximum(0, self.trans_matrix[i_0, i_1] + rho) - np.maximum(0, rho)
            gamma_prime = np.maximum(0, self.trans_matrix[i_0, i_1] + rho_prime) - np.maximum(0, rho_prime)

            self.out_msgs[0] = (self.damp * self.prev_out_msgs[0]) + ((1 - self.damp) * gamma)
            self.out_msgs[1] = (self.damp * self.prev_out_msgs[1]) + ((1 - self.damp) * gamma_prime)

