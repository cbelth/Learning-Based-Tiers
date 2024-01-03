import random
import sigmapie

from phonotactic_model import PhonotacticModel

class TSLIA(PhonotacticModel):
    def __init__(self, 
                 ipa_file='../data/ipa.txt',
                 underspec=True,
                 verbose=True):

        super().__init__(ipa_file=ipa_file, add_segs=False, underspec=underspec, verbose=verbose)

    def train(self, pairs):
        sfs = self.get_sfs(pairs)
        self.tsl = sigmapie.TSL()
        self.tsl.data = sfs
        self.tsl.extract_seginv()
        self.tsl.learn()
        return self

    def score(self, candidates):
        scores = list()
        for cand in candidates:
            scores.append(1 if self.tsl.scan(cand) else 0)
        return scores