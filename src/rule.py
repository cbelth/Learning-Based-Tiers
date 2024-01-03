from utils import UNKNOWN_CHAR
from utils import tolerance_principle
from sequence import Sequence
from segment_inventory import SegmentInventory
from tier import Tier

class Rule:
    '''
    Strict (tier) adjacency rule one of the following forms:
     - A --> [αf] / C_αf __ / del(S)
     - A --> [αf] / __ D_αf / del(S)
    '''
    def __init__(self, target, feats, lc=None, rc=None, tier=None, assimilate=True, seginv=None, underspec=True):
        self.target = target
        self.feats = feats
        if lc is not None and rc is not None:
            raise ValueError('Cannot have both left and right contexts.')
        self.lc = lc
        self.rc = rc
        self.tier = tier if tier else Tier(set(), seginv)
        self.seginv = seginv if seginv else SegmentInventory()
        self.operator = self.seginv.assimilate if assimilate else self.seginv.dissimilate
        self.assimilate = assimilate
        self.underspec = underspec

    def update_c(self, c):
        if self.lc:
            self.update_lc(c)
        else:
            self.update_rc(c)

    def update_lc(self, lc):
        self.lc = lc

    def update_rc(self, rc):
        self.rc = rc

    def __str__(self):
        arrow = '-->' if self.assimilate else '<--'
        if self.lc:
            return f'{self.target} {arrow} {self.feats} / {self.lc if type(self.lc) is not set else tuple(sorted(self.lc))} __ / {self.tier}'
        elif self.rc:
            return f'{self.target} {arrow} {self.feats} / __ {self.rc if type(self.rc) is not set else tuple(sorted(self.rc))} / {self.tier}'            
        else:
            return f'{self.target} {arrow} {self.feats} / __ / {self.tier}'

    def __repr__(self):
        return self.__str__()

    def match(self, window):
        if len(window) == 1: # no context
            return window[0] in self.target
        if self.lc: # left context
            return window[0] in self.lc and window[1] in self.target
        return window[0] in self.target and window[1] in self.rc # right context

    def __len__(self):
        if self.lc or self.rc:
            return 2
        return 1

    def get_changes(self, s):
        '''
        Computes the changes that the rule should make to the sequence.

        :s: a sequence

        :return: a list of (idx, change) tuples, each which is a change to be carried out that the idx position of :s:
        '''
        k = len(self)
        tier_seq = self.tier.project(s)
        
        changes = list()
        left_to_right = self.lc is not None
        tier_ptr = 0 if left_to_right else len(tier_seq) - 1 # a pointer that keeps track of where on the tier we are
        seq_ptr = 0 if left_to_right else len(s) - 1 # a pointer that keeps track of where in the seq we are
        while True:
            seg = s[seq_ptr]
            if seg in tier_seq: # on tier
                # compute window
                if not self.lc and not self.rc: # no context
                    window = Sequence(seg, self.seginv)
                elif self.lc and tier_ptr > 0: # left context
                    window = Sequence(f'{tier_seq[tier_ptr - 1]}{seg}', self.seginv)
                elif self.rc and tier_ptr < len(tier_seq) - 1: # right context
                    window = Sequence(f'{seg}{tier_seq[tier_ptr + 1]}', self.seginv)
                else:
                    window = None
                if window and self.match(window): # check if window matches rule
                    if self.lc: # left context
                        # only assimilate
                        new_seg = self.operator(seg=seg, tgt=window[0], feats=self.feats, only_underspec=self.underspec) # assimilate left
                    elif self.rc: # right context
                        new_seg = self.operator(seg=seg, tgt=window[1], feats=self.feats, only_underspec=self.underspec) # assimilate right
                    else: # no context
                        new_seg = seg # assimilate with self (i.e., no change)
                    changes.append((seq_ptr, new_seg if new_seg else UNKNOWN_CHAR)) # add change to those that need to be made
                    if new_seg:
                        tier_seq[tier_ptr] = new_seg # apply sequentially by updating tier with change too
                tier_ptr += 1 if left_to_right else -1 # move forward on tier
            seq_ptr += 1 if left_to_right else -1
            if left_to_right and seq_ptr == len(s):
                break
            elif seq_ptr == -1:
                break
        return changes

    def apply(self, s):
        '''
        Apply the rule to the string :s:
        '''
        if type(s) is str:
            s = Sequence(s, self.seginv)
        out_tape = s.copy()
        for idx, change in self.get_changes(s):
            out_tape[idx] = change
        return out_tape

    '''
    Call self.apply(s) on Rule objects if used as a callable
    '''
    __call__ = apply

    def get_n_c(self, pairs):
        n, c = 0, 0
        for uf, sf in pairs:
            self.seginv.add_segments_from_str(f'{uf}')
            self.seginv.add_segments_from_str(f'{sf}')
            changes = self.get_changes(uf)
            for idx, change in changes:
                n += 1
                if sf[idx] == change:
                    c += 1
        return n, c

    def get_apps(self, pairs):
        apps = list()
        for uf, _ in pairs:
            changes = self.get_changes(uf)
            apps.extend(list((uf, idx, change) for idx, change in changes))
        return apps

    def accuracy(self, pairs):
        n, c = self.get_n_c(pairs)
        return c / n if n > 0 else 0

    def __eq__(self, other):
        if type(other) is str:
            return self.__str__() == other
        return self.__str__() == other.__str__()

    def __neq__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.__str__())