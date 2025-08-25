from mpl.models.mpl_gurobi_nowait_overlap import MPLGurobiNoWaitOverlap

class MPLGurobiNoWaitNoOverlap(MPLGurobiNoWaitOverlap):
    def __init__(self, data):
        super().__init__(data, overlap=False)