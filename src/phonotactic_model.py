from abc import abstractmethod
from itertools import product as cartesian_product
import numpy as np
import random
from collections import defaultdict

from model import Model
from sequence import Sequence

class PhonotacticModel(Model):
    '''
    A partially-abstract class for Phonotactic models to help them behave more like alternation models.
    '''
    def __init__(self, ipa_file, add_segs, underspec, verbose):
        super().__init__(ipa_file=ipa_file, add_segs=add_segs, underspec=underspec, verbose=verbose)

    @abstractmethod
    def score(self, candidates):
        raise NotImplementedError('AbstractMethod score(candidates) is not implemented for this subclass.')

    def choose(self, uf, opt1, opt2):
        s1, s2 = self.score([opt1, opt2])
        if s1 == s2:
            return random.choice([opt1, opt2])
        return opt1 if s1 > s2 else opt2

    def _get_candidates(self, uf):
        underspec = list(filter(lambda i: f'{uf[i]}' in self.alternating_segs, list(range(len(uf)))))
        underspec_options = dict()
        for un_i in underspec:
            un = uf[un_i]
            underspec_options[un_i] = self.seginv.permute(un, self.alternating_segs[un], only_underspec=self.underspec)

        underspec_options = sorted(underspec_options.items(), key=lambda it: it[0])
        underspec = list(it[0] for it in underspec_options)
        options = list(it[1] for it in underspec_options)
        options_prod = list(cartesian_product(*options))

        cands = list()
        for opts in options_prod:
            assert(len(underspec) == len(opts))
            cand = list(f'{uf}')
            for i, v in zip(underspec, opts):
                cand[i] = f'{v}'
            cand = ''.join(cand)
            cands.append(cand)
        return cands

    def produce(self, uf):
        self.seginv.add_segments_from_str(uf)
        if type(uf) is str:
            uf = Sequence(uf, seginv=self.seginv)
        
        cands = self._get_candidates(uf)
        scored = self.score(cands)
        argmax = np.argmax(scored)
        return cands[argmax]

    def get_sfs(self, pairs):
        self.alternating_segs = defaultdict(set)
        sfs = list()
        for uf, sf in pairs:
            self.seginv.add_segments_from_str(uf)
            self.seginv.add_segments_from_str(sf)
            sfs.append(sf)
            for i in range(len(uf)):
                uf_seg, sf_seg = uf[i], sf[i]
                if uf_seg != sf_seg: # update discrepancy
                    feat_diff = tuple(sorted(self.seginv.feat_diff(uf_seg, sf_seg)))
                    self.alternating_segs[uf_seg].update(feat_diff)
        return sfs