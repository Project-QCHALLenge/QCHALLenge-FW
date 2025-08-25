import pennylane as qml
from pennylane import numpy as np
import time


class SPQAOAnsatz:
    def __init__(self, data, layers = 2, seed = 1, **params):
        self.data = data
        self.layers = layers

        np.random.seed(seed)

        self.lidar_dict = self._create_lidar_dict()
        self.streetpoint_dict = self._create_streetpoint_dict()
        self.n_lidars = len(self.lidar_dict.keys())
        self.max_strpnts_of_lidars = max([len(streatpnts) for streatpnts in self.lidar_dict.values()])
        self.n_streetpoints = len(self.streetpoint_dict.keys())

        self.cost_hamiltonian = self._cost_hamiltonian()

        self.qubit_to_lidar_map = self._create_qubit_to_lidar_map()
        self.strpnt_qubit_map = self._create_qubit_to_strpnt_map()

        self.n_qubits = self.n_lidars + self.max_strpnts_of_lidars
        self.qaoa_device = qml.device(
            "default.qubit", wires=self.n_qubits
        )
        # in the case each lidar should have its own parameter
        """self.parameters = np.random.uniform(
            0, 2 * np.pi, (self.layers, 2, self.n_lidars)
        )"""
        self.parameters = np.random.uniform(
            0, 2 * np.pi, (self.layers, 2)
        )

    def _create_qubit_to_strpnt_map(self):
        """
        Creates a mapping from street points to qubit indices.

        This method generates a dictionary where each key is a street point from 
        `self.streetpoint_dict` and each value is a corresponding qubit index of 
        the quantum cirucit.mThe qubit index is determined by the order of the 
        street points and an offset of `self.n_lidars`.

        Returns:
            dict: A dictionary mapping street points to qubit indices.
        """
        strpnt_qubit_map = {}
        for i, streetpoint in enumerate(self.streetpoint_dict.keys()):
            strpnt_qubit_map[streetpoint] = i + self.n_lidars
        return strpnt_qubit_map

    def _create_qubit_to_lidar_map(self):
        """
        Creates a mapping from lidar identifiers to qubit indices.

        This method generates a dictionary that maps each lidar identifier
        from the `lidar_dict` to a unique qubit index. The indices are assigned
        sequentially based on the order of the lidar identifiers in the dictionary.

        Returns:
            dict: A dictionary where the keys are lidar identifiers and the values
                  are the corresponding qubit indices.
        """
        qubit_to_lidar_map = {}
        for i, lidar in enumerate(self.lidar_dict.keys()):
            qubit_to_lidar_map[lidar] = i
        return qubit_to_lidar_map

    def _create_lidar_dict(self):
        """
        Creates a dictionary mapping lidar nodes to their corresponding edges.

        Returns:
            dict: A dictionary where keys are lidar nodes and values are lists of edges.
        """
        lidar_dict = {}
        for node in self.data.G.nodes:
            # lidar node
            if len(node) == 5:
                edge_list = [
                    sub_tuple
                    for outer_tuple in self.data.G.edges(node)
                    for sub_tuple in outer_tuple
                ]
                lidar_dict[node] = [edge for edge in edge_list if len(edge) == 3]

        return lidar_dict

    def _create_streetpoint_dict(self):
        """
        Creates a dictionary mapping street points to their corresponding edges.

        Returns:
            dict: A dictionary where keys are nodes (with length 3) and values are lists of edges (with length 5).
        """
        streetpoint_dict = {}
        for node in self.data.G.nodes:
            if len(node) == 3:
                edge_list = [
                    sub_tuple
                    for outer_tuple in self.data.G.edges(node)
                    for sub_tuple in outer_tuple
                ]

                streetpoint_dict[node] = [edge for edge in edge_list if len(edge) == 5]
        return streetpoint_dict

    def _initial_state_preparation(self):
        for i in range(self.n_lidars):
            qml.PauliX(wires=i)

    def _cost_hamiltonian(self):
        pauli_z = []
        for lidar in range(len(self.lidar_dict.keys())):
            pauli_z.append(qml.PauliZ(wires=lidar))

        return qml.Hamiltonian([-1.0 for _ in range(len(pauli_z))], pauli_z)

    def _cost_hamiltonian_layer(self, parameter):
        for lidar in range(len(self.lidar_dict.keys())):
            qml.RZ(phi=parameter, wires=lidar)

    def _mixer_hamiltonian_layer(self, parameters):
        """
        Applies a mixer Hamiltonian layer to the quantum circuit.
        This method iterates over each lidar and its associated streetpoints to apply controlled operations 
        on the corresponding qubits. It first applies controlled Pauli-X operations on slack qubits, 
        then applies controlled RX rotations on lidar qubits, and finally undoes the Pauli-X operations 
        on the slack qubits.
        Parameters:
        -----------
        parameters : float
            The rotation angle for the RX gate.
        Notes:
        ------
        - The method uses the `qml.ctrl` function to apply controlled operations.
        - The control wires and control values are determined based on the constraints of the streetpoints 
          and the lidar-to-qubit mapping.
        - If a mandatory lidar is found for a streetpoint, the method exits with an error message.
        """
        for lidar_idx, lidar in enumerate(self.lidar_dict.keys()):
            n_streetpoints_for_lidar = len(self.lidar_dict[lidar])
            # iterate over streetpoints for each lidar and apply controlled operation on lidar qubit if it is allowed
            for idx, streetpoint in enumerate(self.lidar_dict[lidar]):

                constr_sub_lidars = self.streetpoint_dict[streetpoint].copy()
                # if len(constr_sub_lidars)==1:
                #     exit(f"mandatory lidar found for streetpoint {streetpoint}")

                # take out the lidar element from the _sub_lidar list if inside
                if lidar in constr_sub_lidars:
                    constr_sub_lidars.remove(lidar)
                
                ctrl_wires = [self.qubit_to_lidar_map[i] for i in constr_sub_lidars]
                ctrl_val = [0 for _ in range(len(ctrl_wires))]

                qml.ctrl(
                op=qml.PauliX,
                control=ctrl_wires,
                control_values=ctrl_val,
                )(wires=self.n_lidars+idx)

            # apply x rotation if applicable with controlled operation on lidar qubit and ctl the slack qubits
            qml.ctrl(
                op=qml.RX,
                control=[self.n_lidars+i for i in range(n_streetpoints_for_lidar)],
                control_values=[0 for _ in range(n_streetpoints_for_lidar)],
            )(parameters, wires=self.qubit_to_lidar_map[lidar])
            
            # go over slack qubits to undo PauliX application on slack qubits
            for idx, streetpoint in enumerate(self.lidar_dict[lidar]):

                # print("streetpoint", streetpoint, "qubit", self.strpnt_qubit_map[streetpoint])

                constr_sub_lidars = self.streetpoint_dict[streetpoint].copy()

                # if len(constr_sub_lidars)==1:
                #     exit(f"mandatory lidar found for streetpoint {streetpoint}")

                # take out the lidar element from the _sub_lidar list if inside
                if lidar in constr_sub_lidars:
                    constr_sub_lidars.remove(lidar)

                ctrl_wires = [self.qubit_to_lidar_map[i] for i in constr_sub_lidars]
            
                ctrl_val = [0 for _ in range(len(constr_sub_lidars))]
                target_wire = self.n_lidars+idx

                qml.ctrl(
                op=qml.PauliX,
                control=ctrl_wires,
                control_values=ctrl_val,
                )(wires=target_wire)

    def state_probabilities(self, state):
        probabilities = np.abs(state) ** 2
        return {
            bin(i)[2:].zfill(self.n_qubits): float(prob)
            for i, prob in enumerate(probabilities)
        }

    def mixer_qaoa_circuit(self, draw=None):
        """
        Constructs and optionally draws the mixer QAOAnsatz circuit.
        This method creates a quantum circuit based on the Quantum Approximate Optimization Ansatz (QAOAnsatz) 
        using the mixer Hamiltonian. It prepares the initial state, applies the mixer Hamiltonian layers, 
        and returns the final quantum state.
        Args:
            draw (bool, optional): If True, returns a drawing of the circuit. Defaults to None.
        Returns:
            qml.state or str: The final quantum state if draw is False, otherwise a string representation 
                              of the circuit diagram.
        """
        @qml.qnode(self.qaoa_device)
        def _mixer_qaoa_circuit():
            
            self._initial_state_preparation()
            for layer in range(self.layers):
                self._mixer_hamiltonian_layer(self.parameters[layer][1])
            return qml.state()

        if draw:
            print(qml.draw(_mixer_qaoa_circuit)())
        else:
            return _mixer_qaoa_circuit()

    def qaoa_circuit(self, parameters, state_vector=0):
        """
        Constructs and executes a QAOAnsatz (Quantum Approximate Optimization Ansatz) circuit.
        Args:
            parameters (list): A list of parameters for the QAOA layers.
            state_vector (int, optional): Determines the return type of the circuit. 
                                          0 returns the expectation value of the cost Hamiltonian,
                                          1 returns the state vector,
                                          2 returns the drawn circuit. Default is 0.
        Returns:
            qml.QNode or str: Depending on the value of state_vector, returns the expectation value,
                              the state vector, or a drawn representation of the circuit.
        """
        @qml.qnode(self.qaoa_device)
        def _qaoa_circuit(_parameters, _state_vector):
            self._initial_state_preparation()
            for layer in range(self.layers):
                self._cost_hamiltonian_layer(_parameters[layer][0])
                self._mixer_hamiltonian_layer(_parameters[layer][1])
            
            if _state_vector == 1:
                return qml.state()
            elif _state_vector == 0:
                return qml.expval(self.cost_hamiltonian)
            elif _state_vector == 2:
                return qml.state()
                
        if state_vector == 2:
            return qml.draw(_qaoa_circuit)(_parameters=parameters, _state_vector=state_vector)
        else:
            return _qaoa_circuit(_parameters=parameters, _state_vector=state_vector)

    def optimise(self, learning_rate, max_iterations, info):
        """
        Optimizes the parameters of the QAOAnsatz circuit using the Adam optimizer.
        Args:
            learning_rate (float): The learning rate for the Adam optimizer.
            max_iterations (int): The maximum number of iterations for the optimization process.
        Returns:
            tuple: A tuple containing:
                - param_list (list): A list of parameter sets for each iteration.
                - cost_list (list): A list of cost values for each iteration.
                - total_time (float): The total time taken for the optimization process in seconds.
        """
        start_time = time.time()
        param_list = []
        cost_list = []

        def wrapper(parameters):
            self.parameters = parameters
            return self.qaoa_circuit(self.parameters, state_vector=0)
        
        opt = qml.AdamOptimizer(learning_rate)
        for it in range(max_iterations):
            self.parameters = opt.step(wrapper, self.parameters)
            param_list.append(self.parameters)
            cost_list.append(wrapper(self.parameters))
            if info and it % 10 == 0:
                print(f"Step {it}, Cost: {self.qaoa_circuit(self.parameters)}")
        total_time = time.time() - start_time
        return param_list, cost_list, total_time

    def best_solution(self):
        """
        Finds and returns the best solution from the QAOAnsatz circuit.

        This method executes the QAOAnsatz circuit with the given parameters to obtain the state vector.
        It then calculates the probabilities of each state and returns a dictionary of states sorted
        by their probabilities in descending order.

        Returns:
            dict: A dictionary where keys are states and values are their corresponding probabilities,
                  sorted in descending order of probability.
        """
        best_state = self.qaoa_circuit(parameters=self.parameters, state_vector=True)
        prob_dict = self.state_probabilities(best_state)
        return dict(sorted(prob_dict.items(), key=lambda item: item[1], reverse=True))
        
    def allowed_states(self, prob_dict):
        """
        Determines the allowed states based on the given probability dictionary and 
        the constraints defined by street points and lidar mappings.

        Args:
            prob_dict (dict): A dictionary where keys are states (as strings) and values 
                              are their corresponding probabilities.

        Returns:
            tuple: A tuple containing:
                - allowed_states (dict): A dictionary where keys are states and values 
                                         are booleans indicating if the state is allowed.
                - certificated_states (dict): A dictionary where keys are states and values 
                                              are strings indicating the certification status 
                                              of the state.
        """
        allowed_states = {}
        certificated_states = {}
        for state, prob in prob_dict.items():
            adj_prob = round(prob, 5)
            state_allowed_list = []
            for streetpoint in self.streetpoint_dict.keys():
                constr_lidars = self.streetpoint_dict[streetpoint]
                constr_strpnt_qubtis = [
                    self.qubit_to_lidar_map[i] for i in constr_lidars
                ]
                constr_strpnt_qubtis = [int(i) for i in constr_strpnt_qubtis]
                on_lidar_list = [False if state[idx]=="0" else True for idx in constr_strpnt_qubtis]
                if any(on_lidar_list):
                    streetpoint_allowed = True
                    state_allowed_list.append(streetpoint_allowed)
                else:
                    streetpoint_allowed = False
                    state_allowed_list.append(streetpoint_allowed)
            allowed = all(state_allowed_list)

            if (allowed and adj_prob > 0):
                certificated_states[state] = "T-(A,>0)"
                allowed_states[state] = True
            elif (not allowed and adj_prob == 0):
                certificated_states[state] = "T-(NA,=0)"
                allowed_states[state] = True
            elif (allowed and adj_prob == 0):
                certificated_states[state] = "F-(A,=0)"
                allowed_states[state] = False
            elif (not allowed and adj_prob > 0):
                certificated_states[state] = "F-(NA,>0)"
                allowed_states[state] = False

        return allowed_states, certificated_states

    def solve(self, learning_rate, iterations, info=False, **params):
        params_list, cost_list, runtime = self.optimise(learning_rate=learning_rate, max_iterations=iterations, info=info)
        solution_dict = self.best_solution()
        best_solution = list(solution_dict.keys())[0]

        return {"solution": self.qubit_to_lidar(best_solution), "runtime": runtime, "energy": None}

    def qubit_to_lidar(self, answer):
        # created to work in an analog fashion to self.__inverter_matrix() for the plotting function
        reverse_dict = {v: k for k, v in self.qubit_to_lidar_map.items()}
        answer = answer[:len(reverse_dict)]
        sol_dict = {}
        for idx, val in enumerate(answer):
            _lidar = reverse_dict[idx]
            sol_dict[f"x_{_lidar[0]}_{_lidar[1]}_{_lidar[2]}_{_lidar[3]}_{_lidar[4]}"]= int(val)

        return sol_dict