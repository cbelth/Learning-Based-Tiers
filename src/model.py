from abc import ABC, abstractmethod

from segment_inventory import SegmentInventory

class Model:
    '''
    A partially-abstract class for learning Models.
    '''
    def __init__(self, ipa_file='../data/ipa.txt', underspec=True, add_segs=False, seginv=None, verbose=True):
        self.ipa_file = ipa_file
        self.underspec = underspec
        self.add_segs = add_segs
        self.verbose = verbose
        self.seginv = SegmentInventory(ipa_file=ipa_file, add_segs=add_segs)
        if seginv:
            self.seginv = seginv

    @abstractmethod
    def train(self, pairs):
        raise NotImplementedError('AbstractMethod train(pairs) is not implemented for this subclass.')

    @abstractmethod
    def choose(self, uf, opt1, opt2):
        raise NotImplementedError('AbstractMethod choose(uf, opt1, opt2) is not implemented for this subclass.')

    @abstractmethod
    def produce(self, uf):
        raise NotImplementedError('AbstractMethod produce(uf) is not implemented for this subclass.')

    def accuracy(self, test, return_errors=False):
        errors = list()
        t, c, = 0, 0
        is_path = type(test) is str
        if is_path:
            iterable = open(test, 'r')
        else:
            iterable = test
        for line in iterable:
            if is_path:
                uf, sf = line.strip().split('\t')
            else:
                uf, sf = line
            pred = self.produce(uf)
            if sf == pred:
                c += 1
            elif return_errors:
                errors.append((uf, pred, sf))
            t += 1
        if return_errors:
            return c / t if t > 0 else 0, errors
        return c / t if t > 0 else 0