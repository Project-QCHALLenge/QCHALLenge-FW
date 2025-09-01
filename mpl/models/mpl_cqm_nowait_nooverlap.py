from mpl.models.mpl_cqm_nowait_overlap_ import MPLCQMNoWaitOverlap


class MPLCQMNoWaitNoOverlap(MPLCQMNoWaitOverlap):
    def __init__(self, data):
        super().__init__(data, overlap=False)