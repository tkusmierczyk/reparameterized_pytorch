import normflows as nf


class NormFlowWrapper(nf.NormalizingFlow):
    def sample(self, batchSize, D=None, calculate_nll=False):
        x, log_prob = super().sample(batchSize)
        if D:
            x = x.reshape(-1, D)
        return (x, -log_prob) if calculate_nll else x
