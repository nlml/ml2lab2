import numpy as np
class Node(object):
    """
    Base-class for Nodes in a factor graph. Only instantiate sub-classes of Node.
    """
    def __init__(self, name):
        # A name for this Node, for printing purposes
        self.name = name
        # Neighbours in the graph, identified with their index in this list.
        # i.e. self.neighbours contains neighbour 0 through len(self.neighbours) - 1.
        self.neighbours = []
        # Reset the node-state (not the graph topology)
        self.reset()
    def reset(self):
        # Incoming messages; a dictionary mapping neighbours to messages.
        # That is, it maps  Node -> np.ndarray.
        self.in_msgs = {}
        # A set of neighbours for which this node has pending messages.
        # We use a python set object so we don't have to worry about duplicates.
        self.pending = set([])
        self.finished_pend = False
    def add_neighbour(self, nb):
        self.neighbours.append(nb)
    def send_sp_msg(self, other):
        # To be implemented in subclass.
        raise Exception('Method send_sp_msg not implemented in base-class Node')
    def send_ms_msg(self, other):
        # To be implemented in subclass.
        raise Exception('Method send_ms_msg not implemented in base-class Node')
    def receive_msg(self, other, msg):
        # Store the incomming message, replacing previous messages from the same node
        self.in_msgs[other] = msg
        # print msg
        #print self.name, "msg: ", msg, "\n"
        # TODO: add pending messages
        # self.pending.update(...)
        if not self.finished_pend:
            if not self.pending:
                excl_index = self.neighbours.index(other)
                idxs_excluding_other = range(0, excl_index) + range(excl_index + 1, len(self.neighbours))
                for ind in idxs_excluding_other:
                    self.pending.add(self.neighbours[ind])
        if other in self.pending:
            self.pending.remove(other)
            if not self.pending:
                self.finished_pend = True
    def __str__(self):
        # This is printed when using 'print node_instance'
        return self.name
class Variable(Node):
    def __init__(self, name, num_states):
        """
        Variable node constructor.
        Args:
            name: a name string for this node. Used for printing.
            num_states: the number of states this variable can take.
            Allowable states run from 0 through (num_states - 1).
            For example, for a binary variable num_states=2,
            and the allowable states are 0, 1.
        """
        self.num_states = num_states
        # Call the base-class constructor
        super(Variable, self).__init__(name)
    def set_observed(self, observed_state):
        """
        Set this variable to an observed state.
        Args:
            observed_state: an integer value in [0, self.num_states - 1].
        """
        # Observed state is represented as a 1-of-N variable
        # Could be 0.0 for sum-product, but log(0.0) = -inf so a tiny value is preferable for max-sum
        self.observed_state[:] = 0.000001
        self.observed_state[observed_state] = 1.0
    def set_latent(self):
        """
        Erase an observed state for this variable and consider it latent again.
        """
        # No state is preferred, so set all entries of observed_state to 1.0
        # Using this representation we need not differentiate between observed and latent
        # variables when sending messages.
        self.observed_state[:] = 1.0
    def reset(self):
        super(Variable, self).reset()
        self.observed_state = np.ones(self.num_states)
    def marginal(self, Z=None):
        """
        Compute the marginal distribution of this Variable.
        It is assumed that message passing has completed when this function is called.
        Args:
            Z: an optional normalization constant can be passed in. If None is passed, Z is computed.
        Returns: marginal, Z. The first is a numpy array containing the normalized marginal distribution.
         Z is either equal to the input Z, or computed in this function (if Z=None was passed).
        """
        # The marginal, or variable belief, is just the product of all messages coming into that variable
        marg = np.multiply.reduce(np.array([self.in_msgs[nb] for nb in self.neighbours]), 0)
        if Z == None:
            Z = np.sum(marg)
        return marg, Z
    def send_sp_msg(self, other):
        # ... maybe need to add extra controls for leaf nodes...
        # for each of this variable's neighbouring factors (except this factor)
        # take the product of input messages to those variables from their other factors
        # if other is not waiting for you
        if not (self in other.pending or not other.pending and not other.finished_pend):
            return
        # if you are not ready
        if not (not self.pending or (other in self.pending and len(self.pending) == 1)):
            return
        #print self.name, "- send to -", other.name
        excl_index = self.neighbours.index(other)
        idxs_excluding_other = range(0, excl_index) + range(excl_index + 1, len(self.neighbours))
        # leaf variable
        if not idxs_excluding_other:
            msg = [1., 1.]
        else:
            started = False
            for ind in idxs_excluding_other:
                var_excluding_other = self.neighbours[ind]
                if not started:
                    msg = self.in_msgs[var_excluding_other]
                    started = True
                else:
                    msg *= self.in_msgs[var_excluding_other]
            # instead of this for loop maybe can do something like:
            # msg = np.multiply.reduce(np.array([self.in_msgs[v] for v in self.neighbours[idxs_excluding_other]]), 0)
            # or maybe:
            # msg = np.multiply.reduce(np.array([self.in_msgs[v] for v in self.neighbours]), idxs_excluding_other)
        other.receive_msg(self, msg)
    def send_ms_msg(self, other):
        # ... maybe need to add extra controls for leaf nodes...
        # for each of this variable's neighbouring factors (except this factor)
        # take the product of input messages to those variables from their other factors
        # if other is not waiting for you
        if not (self in other.pending or not other.pending and not other.finished_pend):
            return
        # if you are not ready
        if not (not self.pending or (other in self.pending and len(self.pending) == 1)):
            return
        #print self.name, "- send to -", other.name
        excl_index = self.neighbours.index(other)
        idxs_excluding_other = range(0, excl_index) + range(excl_index + 1, len(self.neighbours))
        # leaf variable
        if not idxs_excluding_other:
            msg = [0., 0.]
        else:
            started = False
            for ind in idxs_excluding_other:
                var_excluding_other = self.neighbours[ind]
                if not started:
                    msg = self.in_msgs[var_excluding_other]
                    started = True
                else:
                    msg += self.in_msgs[var_excluding_other]
            # instead of this for loop maybe can do something like:
            # msg = np.multiply.reduce(np.array([self.in_msgs[v] for v in self.neighbours[idxs_excluding_other]]), 0)
            # or maybe:
            # msg = np.multiply.reduce(np.array([self.in_msgs[v] for v in self.neighbours]), idxs_excluding_other)
        other.receive_msg(self, msg)
        pass
class Factor(Node):
    def __init__(self, name, f, neighbours):
        """
        Factor node constructor.
        Args:
            name: a name string for this node. Used for printing
            f: a numpy.ndarray with N axes, where N is the number of neighbours.
               That is, the axes of f correspond to variables, and the index along that axes corresponds to a value of that variable.
               Each axis of the array should have as many entries as the corresponding neighbour variable has states.
            neighbours: a list of neighbouring Variables. Bi-directional connections are created.
        """
        # Call the base-class constructor
        super(Factor, self).__init__(name)
        assert len(neighbours) == f.ndim, 'Factor function f should accept as many arguments as this Factor node has neighbours'
        for nb_ind in range(len(neighbours)):
            nb = neighbours[nb_ind]
            assert f.shape[nb_ind] == nb.num_states, 'The range of the factor function f is invalid for input %i %s' % (nb_ind, nb.name)
            self.add_neighbour(nb)
            nb.add_neighbour(self)
        self.f = f
    def send_sp_msg(self, other):
        # if other is not waiting for you
        if not (self in other.pending or not other.pending and not other.finished_pend):
            return
        # if other is not waiting for you, but it is not initialized
        # if you are not ready
        if not (not self.pending or (other in self.pending and len(self.pending) == 1)):
            return
        #print self.name, "- send to -", other.name
        # ... maybe need to add extra controls for leaf nodes...
        # this factor's output (probs) * prod over x_i connected to this factor (message from x_i to this factor)
        excl_index = self.neighbours.index(other)
        idxs_excluding_other = range(0, excl_index)+range(excl_index+1, len(self.neighbours))
        # factor leaf node
        if not idxs_excluding_other:
            msg = self.f
        else:
            l = [self.in_msgs[self.neighbours[v]] for v in idxs_excluding_other]
            ix = np.multiply.reduce(np.ix_(*l))
            m_idx = range(ix.ndim)
            msg = np.tensordot(self.f, ix, [idxs_excluding_other, m_idx])
        other.receive_msg(self, msg * other.observed_state)
    def send_ms_msg(self, other):
        # if other is not waiting for you
        if not (self in other.pending or not other.pending and not other.finished_pend):
            return
        # if other is not waiting for you, but it is not initialized
        # if you are not ready
        if not (not self.pending or (other in self.pending and len(self.pending) == 1)):
            return
        excl_index = self.neighbours.index(other)
        idxs_excluding_other = range(0, excl_index)+range(excl_index+1, len(self.neighbours))
        # factor leaf node
        if not idxs_excluding_other:
            msg = np.log(self.f)
        else:
            msg = np.log(self.f)
            l = np.sum([self.in_msgs[self.neighbours[v]] for v in idxs_excluding_other])
            msg += l
            msg = np.amax(msg, axis = tuple(idxs_excluding_other))
        other.receive_msg(self, msg * other.observed_state)
        
v_flu = Variable("Influenza", 2)
v_throat = Variable("SoreThroat", 2)
v_fever = Variable("Fever", 2)
v_bronchitis = Variable("Bronchitis", 2)
v_coughing = Variable("Coughing", 2)
v_wheezing = Variable("Wheezing", 2)
v_smokes = Variable("Smokes", 2)
f_flu = Factor("Influenza-prior", np.array([0.05, 1-0.05]), [v_flu])
f_smokes = Factor("Smokes-prior", np.array([0.2, 1-0.2]), [v_smokes])
probs = np.zeros((2,2))
probs[1,1] = 0.3
probs[1,0] = 0.001
probs[0,1] = 1 - probs[1,1]
probs[0,0] = 1 - probs[1,0]
f_throat_flu = Factor("SoreThroat|Influenza", probs, [v_throat, v_flu])
probs = np.zeros((2,2))
probs[1,1] = 0.9
probs[1,0] = 0.05
probs[0,1] = 1 - probs[1,1]
probs[0,0] = 1 - probs[1,0]
f_fever_flu = Factor("Fever|Influenza", probs, [v_fever, v_flu])
probs = np.zeros((2,2,2))
probs[1,1,1] = 0.99
probs[1,1,0] = 0.9
probs[1,0,1] = 0.7
probs[1,0,0] = 0.0001
probs[0,1,1] = 1 - probs[1,1,1]
probs[0,1,0] = 1 - probs[1,1,0]
probs[0,0,1] = 1 - probs[1,0,1]
probs[0,0,0] = 1 - probs[1,0,0]
f_bronch_flu_smokes = Factor("Bronchitis|Influenza,Smokes", probs, [v_bronchitis, v_flu, v_smokes])
probs = np.zeros((2,2))
probs[1,1] = 0.8
probs[1,0] = 0.07
probs[0,1] = 1 - probs[1,1]
probs[0,0] = 1 - probs[1,0]
f_coughing_bronch = Factor("Coughing|Bronchitis", probs, [v_coughing, v_bronchitis])
probs = np.zeros((2,2))
probs[1,1] = 0.6
probs[1,0] = 0.01
probs[0,1] = 1 - probs[1,1]
probs[0,0] = 1 - probs[1,0]
f_wheezing_bronch = Factor("Wheezing|Bronchitis", probs, [v_wheezing, v_bronchitis])
variables = [v_flu, v_throat, v_fever, v_bronchitis, v_coughing, v_wheezing, v_smokes]
factors = [f_flu, f_smokes, f_throat_flu, f_fever_flu, f_bronch_flu_smokes, f_coughing_bronch, f_wheezing_bronch]

def sum_product(node_list):
    for node_ind in range(0, len(node_list)-1):
        node = node_list[node_ind]
        for neighbour in node.neighbours:
            node.send_sp_msg(neighbour)
    for node in reversed(node_list):
        for neighbour in node.neighbours:
            node.send_sp_msg(neighbour)
            
def max_sum(node_list):
    for node_ind in range(0, len(node_list)-1):
        node = node_list[node_ind]
        for neighbour in node.neighbours:
            node.send_ms_msg(neighbour)
    for node in reversed(node_list):
        for neighbour in node.neighbours:
            node.send_ms_msg(neighbour)
# variables = [v_flu, v_throat, v_fever, v_bronchitis, v_coughing, v_wheezing, v_smokes]
# factors = [f_flu, f_smokes, f_throat_flu, f_fever_flu, f_bronch_flu_smokes, f_coughing_bronch, f_wheezing_bronch]
f_flu.pending.add(v_flu)
f_smokes.pending.add(v_smokes)
v_coughing.pending.add(f_coughing_bronch)
v_wheezing.pending.add(f_wheezing_bronch)
v_throat.pending.add(f_throat_flu)
v_fever.pending.add(f_fever_flu)

node_l = [f_flu, v_throat, v_fever, f_throat_flu, f_fever_flu, v_flu, v_coughing, f_coughing_bronch, v_wheezing, f_wheezing_bronch, v_bronchitis, f_bronch_flu_smokes, v_smokes, f_smokes]
def run_and_print(node_l, obs, ms = False):
    for n in node_l:
        n.reset()
    f_flu.pending.add(v_flu)
    f_smokes.pending.add(v_smokes)
    v_coughing.pending.add(f_coughing_bronch)
    v_wheezing.pending.add(f_wheezing_bronch)
    v_throat.pending.add(f_throat_flu)
    v_fever.pending.add(f_fever_flu)
    if obs != None:
        v_flu.set_observed(obs)
    else:
        v_flu.set_latent()
    if ms:
        max_sum(node_l)
    else:
        sum_product(node_l)
    for v in variables:
        print v
        marg, Z = v.marginal()
        print marg / Z
run_and_print(node_l, None, False)
run_and_print(node_l, None, True)
run_and_print(node_l, 1, True)
#run_and_print(node_l, 0)