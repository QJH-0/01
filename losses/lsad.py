class LSADLoss:
    def __init__(self, distill_block_indices=(7, 15, 23), *args, **kwargs):
        self.distill_block_indices = tuple(distill_block_indices)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("LSAD loss is reserved for E6 implementation.")
