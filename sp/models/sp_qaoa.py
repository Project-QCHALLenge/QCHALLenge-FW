import itertools

from sp.models.sp_qubo_onehot import QuboSPOnehot
from sp.models.sp_qubo_binary import QuboSPBinary

from pennylane import numpy as np

import pennylane as qml
from pennylane import qaoa
from typing import Union

import time


class QAOA_SP:
    """
    QAOA / QAOAnsatz class for the sensor positioning problem.

    args:
        - qubo
        - layers of the QAOA algorithm
        - type (binary, onehot, QAOA)
    """
    def __init__(self, QuboModel: Union[QuboSPOnehot, QuboSPBinary], layers = 2, type=None, **params):
        self.QuboModel = QuboModel
        self.layers = layers
        self.type = type

        self.n_lidars = len(self.QuboModel.usedLidars)
        self.n_qubits = np.shape(self.QuboModel.model)[0]
        self.n_slack_qubits = self.n_qubits - self.n_lidars

        # define cost hamiltonian
        self.cost_hamiltonian = self.qubo_to_ising(self.QuboModel.model)
        self._slackbits()

        if type == None:
            # normal QAOA does need a lidar mixer, as the lidar qubits are mixed by the generic QAOA mixer
            self.mixer_hamiltonian = qaoa.x_mixer(wires=range(self.n_qubits))
        elif type == "binary":
            self.mixer_hamiltonian = self._binary_mixer()
            self.lidar_mixer = self._lidar_mixer()
        elif type == "onehot":
            self.mixer_hamiltonian = self._onehot_mixer()
            self.lidar_mixer = self._lidar_mixer()
        else:
            exit("Wrong type")

    def qubo_to_ising(self, qubo):
        """
        create an ising hamiltonian from the binary qubo representation

        variable coefficients are respected, however constant values
        arising from the transformation are ignored
        """
        operator_terms = []
        coefficients = []

        for i, j in itertools.product(range(self.n_qubits), range(self.n_qubits)):

            if qubo[i, j] == 0:
                continue
            # linear terms
            # s_i*s_i wird mit (I_i-Z_i)*(I_i-Z_i)=(I_i-Z_i), I_i ist ein konstanter term
            if i == j:
                operator_terms.append(qml.PauliZ(i))
                coefficients.append(-qubo[i, j])

            #quadratic terms
            # s_i*s_j wird mit (I_i-Z_i)*(I_j-Z_j)=(I_i*I_j-Z_i*I_j-I_i*Z_j+Z_i*Z_j), I_i*I_j ist ein konstanter term
            else:
                operator_terms.append(qml.PauliZ(i) @ qml.PauliZ(j))
                coefficients.append(qubo[i, j] / 2)

                operator_terms.append(qml.PauliZ(i))
                coefficients.append(-qubo[i, j] / 2)

                operator_terms.append(qml.PauliZ(j))
                coefficients.append(-qubo[i, j] / 2)

        return qml.Hamiltonian(coefficients, operator_terms)

    def _slackbits(self):
        """
        counts the slack numbers that are needed per streetpoint
        """
        self.slackbits = []
        if self.type == "binary":
            for s in self.QuboModel.data.G.nodes:
                if len(s) == 3:
                    self.slackbits.append(
                        self.QuboModel._QuboSPBinary__needed_bitnum(
                            len(self.QuboModel.data.G.adj[s].items())
                        )
                    )
        elif self.type == "onehot":
            for s in self.QuboModel.data.G.nodes:
                if len(s) == 3:
                    self.slackbits.append(len(self.QuboModel.data.G.adj[s].items()))

    def _check_one_hot_string(self, string):
        "checks the incoming strings for one-hot constraint"
        def one_hot_check(string):
            #check if all values sum to one
            if sum([int(i) for i in string]) == 1:
                return True
            else:
                return False

        #ignore lidars
        string_2_check = string[self.n_lidars :]
        checked_strings = []

        start = 0
        for slackbit in self.slackbits:
            checked_strings.append(
                one_hot_check(string_2_check[start : start + slackbit])
            )
            start += slackbit

        return all(checked_strings)

    def check_one_hot_strings(self, check_dict=None):
        """
        check the entries of a dictionary for one-hot constraints

        if check_dict=None the sorted_dict which results from the evaluation after an optimisation
        otherwise an arbitrary dictionary can be pased
        """
        tot_checked_strings = []

        if check_dict is None:
            _check_dict = self.sorted_dict.items()
        else:
            _check_dict = check_dict.items()

        for key, val in _check_dict:
            if val == 0:
                continue
            _checked_strings = self._check_one_hot_string(string=key)

            if _checked_strings:
                tot_checked_strings.append(True)
            else:
                tot_checked_strings.append(False)

        return all(tot_checked_strings)

    def _binary_mixer(self):
        """
        create Pauli-X string as the conserving hamiltonian for the QAOAnsatz of the binary encoding

        only the slackvariable qubits are affected by this.

        Although the general problem formulation does not allow for zero slack variables in the case that it can arise
        the QAOA ansatz does not make sense and returns exit()
        """
        X_hamiltonian = [
            qml.PauliX(self.n_lidars + i) for i in range(self.n_slack_qubits)
        ]
        if self.n_slack_qubits == 0:
            exit("cant create mixer")

        coeffs = [1.0] * len(
            X_hamiltonian
        )  # Coefficients for each term in the Hamiltonian

        return qml.Hamiltonian(coeffs, X_hamiltonian)

    def _lidar_mixer(self):
        # creates the lidar mixer which is the one-hot encoding of binary strings. (all strings are feasible)
        # Pauli-X string
        X_hamiltonian = [qml.PauliX(i) for i in range(self.n_lidars)]

        # Coefficients for each term in the Hamiltonian
        coeffs = [1.0] * len(
            X_hamiltonian
        )

        return qml.Hamiltonian(coeffs, X_hamiltonian)

    def _onehot_mixer(self):
        """
        create the conserving hamiltonian for the QAOAnsatz of the one hot encoding. Each subset of slack variables has
        its individual hamiltonian. (the hamiltonian preserves the hamming weight and therefore also one-hot encoding)

        the coefficient is set to 2**(n_qubits) in order to ensure that there is no drastic norm increase
        (however the norm of the state still changes)
        """
        hamiltonians = []
        tot_slack = 0
        for slack in self.slackbits:
            gates = [
                qml.PauliX(self.n_lidars + tot_slack + i)
                @ qml.PauliX(self.n_lidars + tot_slack + (i + 1) % slack)
                for i in range(slack)
            ]
            gates += [
                qml.PauliY(self.n_lidars + tot_slack + i)
                @ qml.PauliY(self.n_lidars + tot_slack + (i + 1) % slack)
                for i in range(slack)
            ]
            hamiltonians.append(qml.Hamiltonian([float(1/2**((self.n_qubits)))] * len(gates), gates))
            tot_slack += slack

        return hamiltonians

    def _qubo_cost(self, x):
        # compute the qubo cost of an array x
        return np.dot(x, np.dot(self.QuboModel.model, x))

    def _xHx_cost(self, state):
        # compute the expectation value of the quantum circuit given an input state
        @qml.qnode(qml.device("default.qubit", wires=self.n_qubits))
        def circuit(_state):
            # Initialize the state to x by applying Pauli-X gates where needed
            for i in range(self.n_qubits):
                if _state[i] == 1:
                    qml.PauliX(wires=i)
            return qml.expval(self.cost_hamiltonian)

        return circuit(_state=state)

    def compute_min_qubo(self):
        #compute the minimum binary state for the given qubo
        binary_states = list(itertools.product([0, 1], repeat=self.n_qubits))
        _binary_states = {}
        # Compute the cost for each binary state and store it
        for state in binary_states:
            if self.type == "onehot":
                if not self._check_one_hot_string(string=state):
                    continue
                print(f"state {state} is  one hot")
            x = np.array(state)
            cost = self._qubo_cost(x)
            _binary_states[state] = float(cost)

        return dict(
                sorted(_binary_states.items(), key=lambda item: item[1], reverse=False)
            )

    def eval_dict_qubo(self):
        # calculate the qubo value for each state in the sorted dictionar (the dictionary evaluated after the last
        # optimisation step )
        bin_state_costs = {}
        for state, n_measurements in self.sorted_dict.items():
            x = np.array([int(i) for i in state], dtype=np.float64)
            bin_state_costs[state] = (self._qubo_cost(x), n_measurements)

        return bin_state_costs

    def eval_dict_xHx(self):
        # calculate the quantum circuit evaluation for each state in the sorted dictionar (the dictionary evaluated
        # after the last optimisation step)

        expval_dict = {}
        N_measur = 0
        for binary_state, n_measurements in self.sorted_dict.items():
            x = np.array([int(i) for i in binary_state], dtype=np.float64)
            expval_dict[binary_state] = (float(self._xHx_cost(x)), n_measurements)
            N_measur += n_measurements

        return expval_dict

    def compute_min_xHx(self):
        # compute the minimum computational basis state of the quantum circuit associated with our input qubo
        binary_states = list(itertools.product([0, 1], repeat=self.n_qubits))

        # dictionary to store the expectation values for each state
        _binary_states = {}
        # Compute the expectation value for each binary state
        for state in binary_states:
            if self.type == "onehot":
                if not self._check_one_hot_string(string=state):
                    continue
            exp_val = self._xHx_cost(state)
            _binary_states[state] = float(exp_val)

        return dict(
                sorted(_binary_states.items(), key=lambda item: item[1], reverse=False)
            )

    def cost_mixer_layer(self, parameters):
        """
        A layer of a single cost and mixer unitaries for each three instances of QAOA, binary and onehot QAOAnsatz
        """
        qaoa.cost_layer(parameters[0], self.cost_hamiltonian)

        #mixer layers
        if self.type ==None:
            qaoa.mixer_layer(parameters[1], self.mixer_hamiltonian)

        elif self.type == "onehot":
            qaoa.mixer_layer(parameters[1], self.lidar_mixer)

            for idx in range(len(self.slackbits)):
                # in one hot qml.exp is used as qaoa.mixer layer uses ApproxTimeEvolution which is not exact for non
                # commuting hamitlonians
                qml.exp(op=self.mixer_hamiltonian[idx], coeff=parameters[idx + 2])

                # qml.ApproxTimeEvolution(hamiltonian=self.mixer_hamiltonian[idx], time=parameters[idx + 2], n=3)
                #qaoa.mixer_layer(parameters[idx + 2], self.mixer_hamiltonian[idx])

        elif self.type == "binary":
            # liad mixer
            qaoa.mixer_layer(parameters[1], self.lidar_mixer)
            # slack variable mixer
            qaoa.mixer_layer(parameters[2], self.mixer_hamiltonian)
            # qml.ApproxTimeEvolution(hamiltonian=self.mixer_hamiltonian, time=parameters[2], n=3)

    def qaoa_circuit(self, params):
        """
        QAOA (QAOAnsatz) circuit including initial state preparation, cost/mixer layers
        """

        #initial state
        if self.type == None or self.type == "binary":
            for i in range(self.n_qubits):
                qml.Hadamard(i)

        elif self.type == "onehot":
            for i in range(self.n_lidars):
                qml.Hadamard(i)

            extra = 0
            for i in self.slackbits:
                hot_encoded_qubit = self.n_lidars + i + extra - 1
                qml.X(hot_encoded_qubit)
                extra += i

        for l in range(self.layers):
            self.cost_mixer_layer(params[l])

    def draw_circuit(self):
        # draw circuit using params
        print(qml.draw(self.qaoa_circuit)((self.params)))

    def state_function(self, init_params):
        """
        function which creates a qnode that returns the state-vector of the quantum circuit executed
        """
        dev_state = qml.device("default.qubit", wires=self.n_qubits)
        @qml.qnode(dev_state)
        def _cost_function(_init_params):
            self.qaoa_circuit(params=_init_params)
            return qml.state()

        return _cost_function(_init_params=init_params)

    def cost_function(self, init_params):
        """
        function which creates a qnode that returns the expectation value of the cost_hamiltonian operator of the
        quantum circuit executed
        """
        dev_cost = qml.device("default.qubit", wires=self.n_qubits)
        @qml.qnode(dev_cost)
        def _cost_function(_init_params):
            self.qaoa_circuit(params=_init_params)
            return qml.expval(self.cost_hamiltonian)

        return _cost_function(_init_params=init_params)

    def eval_function(self, params):

        """
        function which creates a qnode that returns the state-vector of the quantum circuit executed, calculates the
        probabilites and returns the dictionary of states and their probabilites

        used to be qml.counts but had to be manually calculated since qml.counts does not work with unnormalised states
        """
        dev_eval = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(dev_eval)
        def _eval_function(_params):
            self.qaoa_circuit(params=_params)
            return qml.state()

        norm_state = np.abs(_eval_function(_params=params)) ** 2
        check_dict = {
            bin(i)[2:].zfill(self.n_qubits): float(prob) for i, prob in enumerate(norm_state)
        }
        return check_dict

    def most_prob_sol(self, parameters):
        """
        evaluate the most probable solution after an initial optimisation phase
        """
        shot_dict = self.eval_function(parameters)

        self.sorted_dict = dict(
            sorted(shot_dict.items(), key=lambda item: item[1], reverse=True)
        )
        # check total prob
        total_prob = sum([i for i in shot_dict.values()])
        # calculate total probability of first 100 most probable states
        sorted_prob = sum(list(self.sorted_dict.values())[:100])
        #print("sorted_prob", f"{round(100*sorted_prob/total_prob, 3)}%")
        # print(f"evaluations = {list(self.sorted_dict.items())[:100]}")
        max_key = max(shot_dict, key=shot_dict.get)
        return {idx_i: int(i) for idx_i, i in enumerate(max_key)}, shot_dict[max_key]/total_prob

    def get_solution(self, last_parameter):
        """
        Return the most probable solution in a readable form using the __inverter_matrix

        Also prints the total confidence in the most probable solution
        """
        most_prob_solution, shot_val = self.most_prob_sol(parameters=last_parameter)
        #print(f"confidence: {round(100*shot_val,3)}%")
        return self.__inverter_matrix(most_prob_solution)

    def __inverter_matrix(self, sample):
        solution_dict = {
            f"x_{self.QuboModel.usedLidars[i][0]}_{self.QuboModel.usedLidars[i][1]}_{self.QuboModel.usedLidars[i][2]}_{self.QuboModel.usedLidars[i][3]}_{self.QuboModel.usedLidars[i][4]}": sample[
                i
            ]
            for i in range(len(self.QuboModel.usedLidars))
        }
        return solution_dict

    def optimize(self, _max_iterations, _optimizer, _learning_rate, _seed, _info):
        """
        Optimisation function

        args:
            - _max_iterations e.g. 200
            - _optimizer "Adam" or "GD"
            - _learning_rate e.g. 0.01
            - _seed e.g. 501

        The function creates initial optimization parameters and optimizes the according quantum circuit for a maximum
        number of iterations, with an optimizer and a set learning rate.
        """
        np.random.seed(_seed)
        # set initial parameters
        if self.type == None:
            self.params = np.random.uniform(
                0, 2 * np.pi, (self.layers, 2), requires_grad=True
            )
        elif self.type == "onehot":
            self.params = np.random.uniform(
                0, 2 * np.pi, (self.layers, 2 + len(self.slackbits)), requires_grad=True
            )
        elif self.type == "binary":
            self.params = np.random.uniform(
                0, 2 * np.pi, (self.layers, 2 + 1), requires_grad=True
            )

        # define optimizer
        if _optimizer == "GD":
            _optimizer = qml.GradientDescentOptimizer(stepsize=_learning_rate)
        elif _optimizer == "Adam":
            _optimizer = qml.AdamOptimizer(stepsize=_learning_rate)
        else:
            exit("Optimizer must be GD or Adam")

        # start optimization
        start_time = time.time()
        cost_list = []
        params_list = []
        for step in range(_max_iterations):
            self.params, cost = _optimizer.step_and_cost(
                self.cost_function, self.params
            )

            cost_list.append(cost)
            params_list.append(self.params)

            if _info and step % 10 == 0:
                # print cost, total prob of the state and adjusted cost based on the prob
                probs = self.eval_function(params=self.params)
                print(f"Step {step}, Cost: {cost}, Adj Cost {cost/sum([i for i in probs.values()])} total prob = {sum([i for i in probs.values()])}")

        runtime = time.time() - start_time
        return params_list, cost_list, runtime

    def solve(self, iterations, optimizer = "Adam", learning_rate = 0.01, seed = 1, info = False, **params):
        """
        Solving the problem includes optimizing the function and evaluating the result.
        """
        params_list, cost_list, runtime = self.optimize(
            _max_iterations=iterations,
            _optimizer=optimizer,
            _learning_rate=learning_rate,
            _seed=seed,
            _info=info
        )

        # evaluate the most probable state after the last optimisation step
        solution_dict = self.get_solution(last_parameter=params_list[-1])
        return {"solution": solution_dict, "runtime": runtime, "energy": None}
        # return {"solution": solution_dict, "runtime": runtime, "params_list": params_list, "cost_list": cost_list, "energy": None}
