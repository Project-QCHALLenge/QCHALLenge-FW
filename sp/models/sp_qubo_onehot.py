from docplex.mp.model import Model
from dwave.system import DWaveSampler, EmbeddingComposite
from dwave.samplers import SimulatedAnnealingSampler
import numpy as np
import time

class QuboSPOnehot: 
    def __init__(self, data, P1 = 1, P2 = 2, P3 = 2) -> None:
        self.data=data
        #Ob jeder Straßenpunkt abgedeckt werden kann wird hier nicht geprüft
        #self.coverage_possible=-1
        self.mod = Model() 
        self.usedLidars=[]
        self.mandatoryLidars=[]
        self.P1=P1
        self.P2=P2
        self.P3=P3
        self.model=self.__compute_QUBO_Matrix_onehot(P1, P2, P3)

        
    def __inverter_matrix(self, sample):
        solution_dict={
            f'x_{self.usedLidars[i][0]}_{self.usedLidars[i][1]}_{self.usedLidars[i][2]}_{self.usedLidars[i][3]}_{self.usedLidars[i][4]}': 
            float(sample[i]) for i in range(len(self.usedLidars))}
        return solution_dict
    
    def solve(self, solve_func, **config):

        start_time = time.time()
        answer = solve_func(Q=self.model, **config)
        solve_time = time.time() - start_time

        solution = self.__inverter_matrix(answer.first.sample)
        info = answer.info
        
        return {"solution": solution, "energy": answer.first.energy, "runtime": solve_time, "info": info}

      
    def solve_dict_dwave(self, num_reads, chain_strength):
        self.myQUBOMatrix=self.__compute_QUBO_Matrix_onehot(self.P1, self.P2, self.P3)
        ds=DWaveSampler()
        sampler=EmbeddingComposite(ds)
        sampleset=sampler.sample_qubo(self.myQUBOMatrix.tolist(), num_reads=num_reads, chain_strength=chain_strength)
        solution_dict_sa=self.__inverter_matrix(sampleset.first.sample)

        return solution_dict_sa

    
    def solve_dict_annealing(self, num_reads, chain_strength):
        self.myQUBOMatrix=self.__compute_QUBO_Matrix_onehot(self.P1, self.P2, self.P3)
        SA_sampler = SimulatedAnnealingSampler()
        sa_res = SA_sampler.sample_qubo(self.myQUBOMatrix, num_reads=num_reads, chain_strength=chain_strength)
        solution_dict_sa = self.__inverter_matrix(sa_res.first.sample)
        solution_dict_sa = {key.replace('m', '-'): value for key, value in solution_dict_sa.items()}

        return solution_dict_sa


    def __compute_QUBO_Matrix_onehot(self, P1, P2, P3): 
        slacksize=0
        
        for s in self.data.G.nodes:
            #Strassenpunkt
            if len(s)==3:
                slacksize+=len(self.data.G.adj[s].items())
                for ls in self.data.G.adj[s].items():
                    self.usedLidars.append(ls[0])
            
                

        #Nur die Lidare, die einen Straßenpunkt covern, sollen in den solver. Schmeisst CPLEX die auch raus?

        self.usedLidars=list(set(self.usedLidars))
        ilist=list(range(len(self.usedLidars)))
        usedLidars_index=dict(zip(self.usedLidars, ilist))


        myQUBOsize=len(self.usedLidars)+slacksize
        myQUBOMatrix=np.zeros([myQUBOsize,myQUBOsize],dtype=float)

        

        #Term 1 Gesamtenergie
        #print("self.usedLidars", self.usedLidars)

        for i in range(0,len(self.usedLidars)): 
            myQUBOMatrix[i,i]=P1

        #Term 2 Gesamtenergie

        #Letzer Index der Lidare in QUBO Matrix, ab dann gehts mit Slackvariablen y weiter
        slackstart=len(self.usedLidars)

        #Schleife über Straßenpunkte
        for s in self.data.G.nodes:
            #Strassenpunkt?
            if len(s)==3:
                j=0
                #Doppelte Schleife über zu Straßenpunkt zugehörige Lidare (ausmultipliziertes Quadrat)
                slackend=slackstart
                #self.coverage_possible=0
                for lj in self.data.G.adj[s].items():    
                    #self.coverage_possible=1
                    j+=1
                    k=0
                    for lk in self.data.G.adj[s].items():    
                        k+=1
                        lidindj=usedLidars_index[lj[0]]
                        lidindk=usedLidars_index[lk[0]]
                        myQUBOMatrix[slackend-1+j,slackend-1+k]+=P2*j*k
                        myQUBOMatrix[lidindj,slackend-1+k]-=P2*k
                        myQUBOMatrix[slackend-1+j,lidindk]-=P2*j
                        myQUBOMatrix[lidindj,lidindk]+=P2
                    #Index hochzählen innerhalb der Lidare eines Straßenpunkts
                    slackstart+=1

        #Term 3 Gesamtenergie

        #Letzer Index der Lidare in QUBO Matrix, ab dann gehts mit Slackvariablen y weiter
        slackstart=len(self.usedLidars)

        #Schleife über Straßenpunkte
        for s in self.data.G.nodes:
            #Strassenpunkt?
            if len(s)==3:
                j=0
                #Doppelte Schleife über zu Straßenpunkt zugehörige Lidare (ausmultipliziertes Quadrat)
                slackend=slackstart
                for lj in self.data.G.adj[s].items():    
                    j+=1
                    k=0
                    for lk in self.data.G.adj[s].items():    
                        k+=1
                        lidindj=usedLidars_index[lj[0]]
                        lidindk=usedLidars_index[lk[0]]
                        myQUBOMatrix[slackend-1+j,slackend-1+k]+=P3
                        
                    myQUBOMatrix[slackend-1+j,slackend-1+j]-=2*P3
                    #Index hochzählen innerhalb der Lidare eines Straßenpunkts
                    slackstart+=1

        return myQUBOMatrix


        