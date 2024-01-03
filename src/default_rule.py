from rule import Rule
from segment_inventory import SegmentInventory
from utils import UNKNOWN_CHAR

class DefaultRule(Rule):
    '''
    A class for the elsewhere/default condition of a (dis)harmony generalization.
    '''
    def __init__(self, target, feats, vals, seginv=None):
        self.target = target
        self.feats = feats
        self.vals = vals
        self.seginv = seginv if seginv else SegmentInventory()

    def __str__(self):
        return f'(default) {self.target} --> {self.vals} {self.feats} / __'

    def __len__(self):
        return 1

    def get_changes(self, s):
        '''
        Computes the changes that the rule should make to the sequence.

        :s: a sequence

        :return: a list of (idx, change) tuples, each which is a change to be carried out that the idx position of :s:
        '''
        k = len(self)
        
        changes = list()
        for seq_ptr in range(len(s)):
            seg = s[seq_ptr]
            if self.match(seg): # check if window matches rule
                new_seg_vec = list(seg.feature_vec)
                for feat, val in zip(self.feats, self.vals):
                    new_seg_vec[self.seginv.feature_space.index(feat)] = val
                new_seg = self.seginv[new_seg_vec]
                changes.append((seq_ptr, new_seg if new_seg else UNKNOWN_CHAR)) # add change to those that need to be made
        return changes
