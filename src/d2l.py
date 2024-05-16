from collections import defaultdict
import numpy as np
import random

from model import Model
from utils import tolerance_principle, LEFT_WORD_BOUNDARY, RIGHT_WORD_BOUNDARY, UNKNOWN_CHAR
from rule import Rule
from natural_class import NaturalClass
from sequence import Sequence
from tier import Tier
from discrepancy import Discrepancy
from default_rule import DefaultRule

class D2L(Model):
    '''
    The code for the paper's proposed model, D2L.
    '''
    def __init__(self, 
                 threshold=tolerance_principle,
                 ipa_file='../data/ipa.txt',
                 underspec=True,
                 add_segs=False,
                 seginv=None,
                 verbose=True):

        super().__init__(ipa_file=ipa_file, 
                         add_segs=add_segs, 
                         underspec=underspec, 
                         seginv=seginv,
                         verbose=verbose)

        self.threshold = threshold
        self.pairs = set()

        self.discrepancy = None
        self.rule = None
        self.default = None

    def run(self):
        '''
        Runs the model after computing the discrepancy and tabulating the training data.
        '''
        if self.discrepancy is None:
            return
        assim_rule = self.build_rule(self.discrepancy, tierset=NaturalClass('*', self.seginv))
        dissim_rule = self.build_rule(self.discrepancy, tierset=NaturalClass('*', self.seginv), assimilate=False)
        if assim_rule and not dissim_rule:
            self.rule = assim_rule
        elif dissim_rule and not assim_rule:
            self.rule = dissim_rule
        elif assim_rule and dissim_rule:
            assim_acc, dissim_acc = assim_rule.accuracy(self.pairs), dissim_rule.accuracy(self.pairs)
            self.rule = assim_rule if assim_acc >= dissim_acc else dissim_rule
        else:
            self.rule = None
            self.default = None

    def train(self, pairs, discrepancy=None):
        '''
        Trains the model.
        '''
        self.pairs = set()
        self.discrepancy = discrepancy
        self.rule = None
        for pair in pairs:
            self.add_incremental(pair, build_discrepancy=discrepancy is None)
        self.run()
        return self

    def choose(self, uf, opt1, opt2):
        '''
        Run 2AFC trial.
        '''
        pred = self.produce(uf)
        if pred == opt1:
            return pred
        if pred == opt2:
            return pred
        return random.choice([opt1, opt2])

    def produce(self, uf):
        '''
        Produce an sf for the input uf
        '''
        self.seginv.add_segments_from_str(uf)
        if type(uf) is str:
            uf = Sequence(uf, seginv=self.seginv)

        sf = uf
        if self.rule:
            sf = self.rule(sf)
        if self.underspec and self.default:
            sf = self.default(sf)
        return sf

    # calling the D2L object amounts to calling its produce() method
    __call__ = produce

    def add_incremental(self, pair, build_discrepancy=True):
        '''
        Tabluate a (uf, sf) pair
        '''
        uf, sf = pair
        self.seginv.add_segments_from_str(uf)
        self.seginv.add_segments_from_str(sf)
        uf, sf = Sequence(uf, self.seginv), Sequence(sf, self.seginv)
        self.pairs.add((uf, sf))

        if build_discrepancy:
            for i in range(len(uf)):
                uf_seg, sf_seg = uf[i], sf[i]
                if uf_seg != sf_seg: # update discrepancy
                    feat_diff = tuple(sorted(self.seginv.feat_diff(uf_seg, sf_seg)))
                    if self.discrepancy is None:
                        self.discrepancy = Discrepancy(feat_diff)
                    self.discrepancy.add(uf, i, uf_seg, sf_seg)
        
        return uf, sf

    def contexts(self, left_right, discrep, tier):
        '''
        Computes the left/right contexts of the target segments on the :tier:

        :left_right: if 'left,' returns left contexts, otherwise (e.g., 'right'), returns right contexts
        '''
        ctxts = set()
        for uf, sf in self.pairs:
            uf_tier, sf_tier = tier.project(uf, sf)
            for i in range(len(uf_tier)):
                uf_tier_seg, sf_tier_seg = uf_tier[i], sf_tier[i]
                if (uf_tier_seg, sf_tier_seg) in discrep:
                    if left_right == 'left': # left contexts
                        if i == 0:
                            ctxts.add(LEFT_WORD_BOUNDARY)
                        else:
                            ctxts.add(uf_tier[i - 1])
                    else: # right contexts
                        if i == len(uf_tier) - 1:
                            ctxts.add(RIGHT_WORD_BOUNDARY)
                        else:
                            ctxts.add(uf_tier[i + 1])
        return ctxts

    def get_best_sep(self, pos, neg):
        '''
        Computes the feature that best separates the :pos: segments, which must be on the tier, from the :neg: segments, which should not be on the tier.
        '''
        pos_feats = self.seginv.shared_feats(pos)

        separate = dict()
        for feat in pos_feats:
            delset = self.seginv.extension_complement({feat})
            if len(neg.difference(delset)) == 0:
                separate[feat] = len(delset) # quality of perf splits is determined by how many segs in delset
        if len(separate) == 0:
            return None
        return sorted(separate.items(), key=lambda it: (it[-1], it[0][1:]))[0][0] # return the feat that leads to smallest delset

    def elsewhere(self, discrep, rule):
        '''
        Computes the default/elsewhere condition, if one works.
        '''
        covered = set(it[:-1] for it in rule.get_apps(self.pairs))
        need_covered = set(it[:-1] for it in discrep.annotated)
        if len(need_covered.difference(covered)) == 0:
            return True
        elif not self.underspec: # can't apply default for fully specified UFs
            return False
        default_options = defaultdict(int)
        for uf, idx, sf_seg in discrep.annotated:
            if (uf, idx) not in covered:
                feat_assignment = tuple(self.seginv.get_val(sf_seg, feat) for feat in discrep.feat_diff)
                default_options[feat_assignment] += 1
        if len(default_options) > 0:
            default = sorted(default_options.items(), reverse=True, key=lambda it: it[-1])[0]
            c, n = default[1], sum(default_options.values())
            if c == n: # As described in Sec 2.3.3, we require no alternation among underextensions. Future work could soften this using the Tolerance Principle
                self.default = DefaultRule(target=set(alt[0] for alt in discrep.alternations), feats=discrep.feat_diff, vals=default[0], seginv=self.seginv)
                return True
        return False

    def build_tierset(self, discrep, tier, assimilate):
        '''
        :return: a set or NaturalClass to be deleted to form a tier.
        '''
        currently_deleted_segs = tier.delset
        adj_that_do_not_work = self.get_adj_that_do_not_work(discrep, tier, assimilate=assimilate)
        pos = list(discrep.get_alternating()) # positive class is alternating segs
        neg = adj_that_do_not_work.union(currently_deleted_segs).difference(pos).difference({UNKNOWN_CHAR}) # negative class is adj segments that don't work with the alternation (del pos b.c. alternating segs must be on tier)
        if len(neg) == 0: # cannot construct a tier when there are no items we know cannot be on the tier
            return None

        best = self.get_best_sep(pos, neg)
        if best is None: # no feature separates pos/neg
            neg_complement = self.seginv.complement(neg).difference({UNKNOWN_CHAR})
            will_not_work = self.get_adj_that_do_not_work(discrep, Tier(neg_complement, self.seginv), assimilate=assimilate).difference(pos)
            while neg_complement != neg_complement.difference(will_not_work):
                neg_complement.difference_update(will_not_work) # exclude those that will not work
                will_not_work = self.get_adj_that_do_not_work(discrep, Tier(neg_complement, self.seginv), assimilate=assimilate).difference(pos)
            return neg_complement # preserve the complement of neg on the tier

        tierset = NaturalClass({best}, self.seginv)
        return tierset

    def build_rule(self, discrep, tierset, depth=0, assimilate=True):
        alternating_segs_ufs = discrep.get_alternating_ufs()
        tier = Tier(tierset, self.seginv) # build a tier

        rs = list()
        # try left contexts
        lcset = self.contexts('left', discrep, tier) # get left contexts
        lr = Rule(target=alternating_segs_ufs, lc=lcset, feats=discrep.feat_diff, tier=tier, assimilate=assimilate, seginv=self.seginv, underspec=self.underspec) # build a rule
        if self.elsewhere(discrep, lr): # rule must cover all alternations
            rs.append(lr)
        # try right contexts
        rcset = self.contexts('right', discrep, tier) # get right contexts
        rr = Rule(target=alternating_segs_ufs, rc=rcset, feats=discrep.feat_diff, tier=tier, assimilate=assimilate, seginv=self.seginv, underspec=self.underspec) # build a rule
        if self.elsewhere(discrep, rr) and len(rcset.difference({LEFT_WORD_BOUNDARY, RIGHT_WORD_BOUNDARY})) != 0: # rule must cover all alternations
            rs.append(rr)

        # choose the best rule
        n_c = dict((r, r.get_n_c(self.pairs)) for r in rs)
        rs = sorted(rs, reverse=True, key=lambda r: n_c[r][1] / n_c[r][0])
        if len(rs) > 0:
            r = rs[0]
            n, c = n_c[r]
            if self.threshold(n=n, c=c):
                ctxt = r.lc if r.lc else r.rc

                # set the ctxt to equal the tier if the ctxt contains at least half the tier segments and doing so does not change the rule's accuracy
                acc_before = r.accuracy(self.pairs)
                r.update_c(r.tier.tierset)
                acc_after = r.accuracy(self.pairs)
                if acc_after < acc_before: # change it back
                    r.update_c(ctxt)

                if self.verbose:
                    print(f'**** Passed: {r} n = {n} c = {c} e = {n - c} <= {round(n / np.log(n), 1)} ****')
                return r

        old_tierset = tierset
        tierset = self.build_tierset(discrep, tier, assimilate=assimilate)
        if tierset is None or old_tierset == tierset: # if no progress was made or no tierset could be constructed (e.g., no neg items), a rule cannot be found at this point
            return None
        return self.build_rule(discrep, tierset, depth=depth+1, assimilate=assimilate) # recurse

    def get_adj_that_do_not_work(self, discrep, tier, assimilate):
        operator = self.seginv.assimilate if assimilate else self.seginv.dissimilate
        delset = set()
        for uf, sf in self.pairs: # iterate over all pairs
            uf_tier, sf_tier = tier.project(uf, sf) # project tier
            for i in range(len(uf_tier)):
                uf_tier_seg, sf_tier_seg = uf_tier[i], sf_tier[i]
                if (uf_tier_seg, sf_tier_seg) in discrep: # if the discrepancy is the one we are trying to account for
                    if i > 0: # left context
                        lc = uf_tier[i - 1] # get left context
                        new_seg = operator(seg=uf_tier_seg, tgt=lc, feats=discrep.feat_diff, only_underspec=self.underspec)
                        failed = not new_seg or new_seg != sf_tier_seg
                        if failed: # if assimilating to lc does not work, then add to delset
                            delset.add(lc)
                    if i < len(uf_tier) - 1: # right context
                        rc = uf_tier[i + 1]
                        new_seg = operator(seg=uf_tier_seg, tgt=rc, feats=discrep.feat_diff, only_underspec=self.underspec)
                        failed = not new_seg or new_seg != sf_tier_seg
                        if failed: # if assimilating to rc does not work, then add to delset
                            delset.add(rc)
        return delset

    def __str__(self):
        if self.rule is None:
            return 'No Rule.'
        return f'{self.rule}'

    def __repr__(self):
        return self.__str__()