from lb import *
from pas import *
from acl import *
from mpl import *
from sp import *
from tr import *
from tl import *



model_classes = {
"PAS": {
        "data": PASData,
        "evaluation": PASEvaluation,
        "plot": PASPlot,
        "cplex_model":PASCplex,
        "decomposition_model":PASDecomposition,
        "gurobi_model":PASGurobi,
        "gurobiconvex_model":PASGurobiConvex,
        "qubo_model":PASQubo
    },
    "ACL": {
        "data": ACLData,
        "evaluation": ACLEvaluation,
        "plot": ACLPlot,
        "cplex_model":ACLCplex,
        "gurobi_model":ACLGurobi,
        "cplexquadratic_model":ACLCplexQuadratic
    },
    "MPL": {
        "data": MPLData,
        "evaluation": MPLEvaluation,
        "plot": MPLPlot,
        "cqmnowaitoverlap_model":MPLCQMNoWaitOverlap,
        "cqmnowaitnooverlap_model":MPLCQMNoWaitNoOverlap,
        "gurobinowaitnooverlapreduced_model":MPLGurobiNoWaitNoOverlapReduced,
        "gurobinowaitnooverlap_model":MPLGurobiNoWaitNoOverlap,
        "gurobinowaitoverlapreduced_model":MPLGurobiNoWaitOverlapReduced,
        "gurobinowaitoverlap_model":MPLGurobiNoWaitOverlap,
        "gurobiwaitoverlap_model":MPLGurobiWaitOverlap
    },
    "SP": {
        "data": SPData,
        "evaluation": SPEvaluation,
        "plot": SPPlot,
        "cplex_model":SPCplex,
        "grover_model":SPGrover,
        "gurobi_model":SPGurobi,
        "heuristic_model":SPHeuristic,
        "qaoa_model":SPQaoa,
        "qaoansatz_model":SPQaoansatz,
        "qubobinary_model":SPQuboBinary,
        "quboonehot_model":SPQuboOnehot,
        "scip_model":SPScip
    },
    "TR": {
        "data": TRData,
        "evaluation": TREvaluation,
        "plot": TRPlot,
        "cplex_model":TRCplex
    },
    "TL": {
        "data": TLData,
        "evaluation": TLEvaluation,
        "plot": TLPlot,
        "cplex_model":TLCplex,
        "gurobi_model":TLGurobi,
        "qubo_model":TLQubo,
        "danzigwolfe_model":TLDanzigWolfe,
        "scip_model":TLScip
    },
    "LB": {
        "data": LBData,
        "evaluation": LBEvaluation,
        "plot": LBPlot,
        "dwavecqmunaryuniformtruckcapacity_model":LBDWAVECQMUnaryUniformTruckCapacity,
        "dwavequbounaryuniformtruckcapacity_model":LBDWAVEQUBOUnaryUniformTruckCapacity,
        "gurobicqmunaryuniformtruckcapacity_model":LBGurobiCQMUnaryUniformTruckCapacity,
        "gurobiqubounaryuniformtruckcapacity_model": LBGurobiQUBOUnaryUniformTruckCapacity
    }
}
