from mpl.models.mpl_gurobi_nowait_overlap_reduced import MPLGurobiNoWaitOverlapReduced

class MPLGurobiNoWaitNoOverlapReduced(MPLGurobiNoWaitOverlapReduced):
    def __init__(self, data):
        super().__init__(data, overlap=False)