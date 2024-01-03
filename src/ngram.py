from collections import defaultdict
import numpy as np
from nltk import ngrams as get_ngrams

from phonotactic_model import PhonotacticModel
from utils import LEFT_WORD_BOUNDARY, RIGHT_WORD_BOUNDARY

class NGram(PhonotacticModel):
    '''
    Baseline NGram model (by default, trigram model).
    '''
    def __init__(self, 
                 n=3,
                 ipa_file='../data/ipa.txt',
                 underspec=True,
                 verbose=True):
        super().__init__(ipa_file=ipa_file, add_segs=True, underspec=underspec, verbose=verbose)
        self.n = n

    def score(self, candidates):
        return list(self.ngram_score(cand) for cand in candidates)

    def train(self, pairs):
        sfs = self.get_sfs(pairs)
        self.segs = set()
        for sf in sfs:
            self.segs.update(sf)

        self.ngram_counts = defaultdict(int)
        for sf in sfs:
            for ngram in self.get_ngrams(sf):
                self.ngram_counts[ngram] += 1
        return self

    def get_ngrams(self, w):
        return list(''.join(ngram) for ngram in get_ngrams(f'{LEFT_WORD_BOUNDARY}{w}{RIGHT_WORD_BOUNDARY}', n=self.n))

    def ngram_score(self, w, smoothing=1.0):
        Z = sum(self.ngram_counts.values())
        num_segs = len(self.segs)

        score = 0
        for ngram in self.get_ngrams(w):
            count = self.ngram_counts[ngram] + smoothing if ngram in self.ngram_counts else smoothing
            log_prob = np.log(count / (Z + num_segs * smoothing))
            score += log_prob
        return score