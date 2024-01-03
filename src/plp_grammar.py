from model import Model
from utils import tolerance_principle
from d2l import D2L
from sequence import Sequence
from discrepancy import Discrepancy

class PLP_Grammar(Model):
    '''
    Runs D2L within the framework of:

    A Learning-Based Account of Local Phonological Processes
    Caleb Belth
    Phonology, In Press.
    '''
    def __init__(self, 
                 threshold=tolerance_principle,
                 ipa_file='../data/ipa.txt',
                 underspec=True,
                 add_segs=False,
                 verbose=True):

        super().__init__(ipa_file=ipa_file, 
                         add_segs=add_segs, 
                         underspec=underspec, 
                         verbose=verbose)

        self.threshold = threshold
        self.pairs = set()

        self.discrepancies = dict()
        self.rules = dict()
        self.defaults = dict()

    def train(self, pairs):
        self.pairs = set()
        self.discrepancies = dict()
        self.rules = dict()
        self.defaults = dict()

        for pair in pairs:
            self.add_incremental(pair)

        for discrep in self.discrepancies:
            d2l = D2L(ipa_file=self.ipa_file, 
                      add_segs=self.add_segs, 
                      underspec=self.underspec, 
                      verbose=self.verbose, 
                      seginv=self.seginv)
            d2l.train(self.pairs, discrepancy=discrep)
            if d2l.rule:
                self.rules[discrep] = d2l.rule
            if d2l.default:
                self.defaults[discrep] = d2l.default

    def produce(self, uf):
        self.seginv.add_segments_from_str(uf)
        if type(uf) is str:
            uf = Sequence(uf, seginv=self.seginv)

        sf = uf
        for discrep in self.discrepancies:
            rule = self.rules[discrep]
            default = self.defaults[discrep]
            if rule:
                sf = rule(sf)
            if self.underspec and default:
                sf = default(sf)
        return sf

    __call__ = produce

    def add_incremental(self, pair):
        uf, sf = pair
        self.seginv.add_segments_from_str(uf)
        self.seginv.add_segments_from_str(sf)
        uf, sf = Sequence(uf, self.seginv), Sequence(sf, self.seginv)
        self.pairs.add((uf, sf))

        for i in range(len(uf)):
            uf_seg, sf_seg = uf[i], sf[i]
            if uf_seg != sf_seg: # update discrepancies
                feat_diff = tuple(sorted(self.seginv.feat_diff(uf_seg, sf_seg)))
                
                discrep = Discrepancy(feat_diff)
                # map to any supersets. E.g., map (back,) to (back,round)
                superset_discreps = list(filter(lambda _discrep: _discrep != discrep \
                                                                 and set(discrep.feat_diff).issubset(_discrep.feat_diff) \
                                                                 and len(set(_discrep.feat_diff).difference(discrep.feat_diff)) == 1, 
                                                self.discrepancies.keys()))
                if len(superset_discreps) == 1:
                    discrep = superset_discreps[0]
                # replace any subsets with this one. E.g., replace (back,) with (back,round)
                subset_discreps = list(filter(lambda _discrep:  _discrep != discrep \
                                                                and set(discrep.feat_diff).issuperset(_discrep.feat_diff) \
                                                                and len(set(discrep.feat_diff).difference(_discrep.feat_diff)) == 1, 
                                            self.discrepancies.keys()))
                if len(subset_discreps) == 1: # replace subsets with this one
                    old = subset_discreps[0]
                    del self.discrepancies[old]
                    del self.rules[old]
                    del self.defaults[old]
                    old.feat_diff = discrep.feat_diff
                    discrep = old
                if discrep not in self.discrepancies:
                    self.discrepancies[discrep] = discrep
                self.discrepancies[discrep].add(uf, i, uf_seg, sf_seg)
                # initialize entries in dicts for this discrepancy
                self.rules[discrep] = None
                self.defaults[discrep] = None
    