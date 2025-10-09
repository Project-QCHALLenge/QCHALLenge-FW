#import pennylane as qml
from sp.data.sp_data import SPData
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, transpile
from numpy import pi
#from qiskit.circuit.library import C4XGate
from qiskit.visualization import circuit_visualization
import itertools as it
from qiskit_aer import AerSimulator, AerProvider, StatevectorSimulator
from qiskit.quantum_info import SparsePauliOp
import math
from qiskit_aer.primitives import EstimatorV2 as Estimator
from qiskit.quantum_info import Statevector, SparsePauliOp
from itertools import combinations, product
import time
from abstract.models.abstract_model import AbstractModel



class GroverSP(AbstractModel):
    def __init__(self, data: SPData) -> None:
        self.data=data
        self.usedLidars=[]

    def solve(self, **config):
         start_time = time.time()
         solution = self.solve_dict(**config)
         solve_time = time.time() - start_time
         return {"solution": solution, "energy": 0, "runtime": solve_time}
       
    def solve_dict(self, sim_kind = 'sample', num_reads=100, num_grover_iterations=0, print_circuit=False):
        lidar_count=len(self.data.listLidar)
        lidar_range=range(lidar_count)
        index_to_lidar_dict={l: self.data.listLidar[i] for i, l in enumerate(lidar_range)}
 #       lidar_to_index_dict=dict((v,k) for k,v in index_to_lidar_dict.items())
        
        self.myCircuit, qc=self.__setup_circuit(num_grover_iterations, print_circuit)
        if sim_kind=='exact': 
#            state = Statevector.from_instruction(qc)
#            return state.to_dict()
            print('Statevector simulator not available!') 
            return -1
        elif sim_kind=='sample': 
            simulator = AerSimulator()
            c = transpile(self.myCircuit, simulator)
            result = simulator.run(c, shots=num_reads, memory=True).result()
            counts = result.get_counts(c)
            num_list=[]
            for key in counts:
                num_list.append(sum([int(i) for i in key]))
            maxcount=max(counts.values())
            mincount=min(counts.values())
            min_numlid=len(self.data.listLidar)+1
            for key in counts:
                numlid = sum([int(i) for i in key])
                if numlid<min_numlid and counts[key]>=(maxcount+mincount)/2: 
                    min_numlid=numlid
                    # little/big endian definition
                    conf=key[::-1]
            sol={}
            j=0
            for var in list(conf):
                new_key='x_'+str(index_to_lidar_dict[j]).replace(" ","_").replace("(","").replace(")","")
                new_key=new_key.replace(",","")
                sol[new_key]=int(var)
                j+=1
            return sol
        else:   
            print('Simulator not known!') 
            return -1

    def build_model(self, num_grover_iterations, print_circuit=False):
        self.__setup_circuit(num_grover_iterations, print_circuit)

    def __setup_circuit(self, num_grover_iterations, print_circuit=False):
        lidar_count=len(self.data.listLidar)
        streetpoint_count=len(self.data.listStreetPoints)
        qreg_q = QuantumRegister(lidar_count+streetpoint_count+1, 'q')
        creg_c = ClassicalRegister(lidar_count, 'c')
        circuit = QuantumCircuit(qreg_q, creg_c)
        lidar_range=range(lidar_count)
        streetpoint_range=range(lidar_count,lidar_count+streetpoint_count )
        oracle_bit=qreg_q[lidar_count+streetpoint_count]
    
        index_to_lidar_dict={l: self.data.listLidar[i] for i, l in enumerate(lidar_range)}
        lidar_to_index_dict=dict((v,k) for k,v in index_to_lidar_dict.items())
        index_to_streetpoint_dict={s: self.data.listStreetPoints[i] for i, s in enumerate(streetpoint_range)}
        streetpoint_to_index_dict=dict((v,k) for k,v in index_to_streetpoint_dict.items())
        # Assume 50% of the configurations are valid
        estimated_portion_of_valid_combination=0.5
        if num_grover_iterations==0: 
            num_grover_iterations = math.floor(pi/4.0*math.sqrt(1/estimated_portion_of_valid_combination)) 
            if num_grover_iterations==0: 
                num_grover_iterations=1

        
        
        # for s in self.data.G.nodes: 
        #     #Staßenpunkt?
        #     if len(s)==3:
        #         for l in self.data.G.adj[s].items():
        #             self.usedLidars.append(l[0])
        # self.usedLidars=list(set(self.usedLidars))

        
        #Prep circuit for grover search
        for i in lidar_range:
            circuit.h(qreg_q[i])
        circuit.x(oracle_bit)
        circuit.h(oracle_bit)
        
        # for i in range(0,qreg_q.size):
        #     circuit.barrier(qreg_q[i])

        
        #Oracle function
        for g in range(num_grover_iterations): 
            #negate bits in order to use toffoli as "or" gate
            for i in it.chain(lidar_range, streetpoint_range):
                circuit.x(qreg_q[i])    
            
            #if at least one adjacent lidar is on, streetpoint is covered
            for s in self.data.G.nodes: 
                #Staßenpunkt?
                if len(s)==3:
                    larr=[]

                    for l in self.data.G.adj[s].items():
                        #Nur die Lidare, die einen Straßenpunkt covern, sollen in den solver. Schmeisst CPLEX die auch raus?
                        larr.append(lidar_to_index_dict[l[0]])
                        self.usedLidars.append(l[0])
                    if len(larr)==0:
                        print('Problem unsolvable!')
                        return -1
                    circuit.mcx(larr,streetpoint_to_index_dict[s],0)
            # if all streetpoints covered, oracle bit is set
            circuit.mcx(list(streetpoint_range), lidar_count+streetpoint_count, 0)
            #reverse streetpoint coverage 
            for s in list(self.data.G.nodes)[::-1]: 
                #Staßenpunkt?
                if len(s)==3:
                    larr=[]

                    for l in list(self.data.G.adj[s].items())[::-1]:
                        #Nur die Lidare, die einen Straßenpunkt covern, sollen in den solver. Schmeisst CPLEX die auch raus?
                        larr.append(lidar_to_index_dict[l[0]])
                    circuit.mcx(larr,streetpoint_to_index_dict[s],0)
            #reverse negation
            for i in it.chain(lidar_range, streetpoint_range):
                circuit.x(qreg_q[i])    

            # for i in range(0,qreg_q.size):
            #     circuit.barrier(qreg_q[i])
            #Diffusion step

            for i in lidar_range:
                circuit.h(qreg_q[i])
            for i in lidar_range:
                circuit.x(qreg_q[i])
            circuit.h(qreg_q[lidar_count-1])

            circuit.mcx(list(range(lidar_count-1)), lidar_count -1, 0)
            
            circuit.h(qreg_q[lidar_count-1])
            for i in lidar_range:
                circuit.x(qreg_q[i])
            for i in lidar_range:
                circuit.h(qreg_q[i])
            
            # for i in range(0,qreg_q.size):
            #     circuit.barrier(qreg_q[i])
           
        
        for i in lidar_range:
            circuit.measure(qreg_q[i],i)


        if print_circuit:    
            print(circuit)
       


        return circuit, QuantumCircuit(qreg_q)
        