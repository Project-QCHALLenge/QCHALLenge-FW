from sp import *

model_classes = {
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
}