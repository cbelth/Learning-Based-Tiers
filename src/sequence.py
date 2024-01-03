from natural_class import NaturalClass
from segment import Segment

class Sequence:
    '''
    A class representing a sequence of Segments.
    '''
    def __init__(self, seq, seginv=None):
        self.seginv = seginv
        if type(seq) == str:
            seq = list(seginv[seg] for seg in seq)
        elif type(seq) == Segment:
            assert(len(seq) == 1)
            seq = [seq]
        self.seq = seq

    def copy(self):
        return Sequence(list(self.seq), self.seginv)

    def __len__(self):
        return len(self.seq)

    def __str__(self):
        s = ''
        for seg in self.seq:
            s += f'{seg}'
        return s

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if type(other) is str:
            return f'{self}' == other
        if type(other) is Segment:
            return f'{self}' == f'{other}'
        if len(self) != len(other):
            return False
        for seg, other_seg in zip(self.seq, other if type(other) is not Sequence else other.seq):
            if seg != other_seg:
                return False
        return True

    def __neq__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        return self.__str__() < other.__str__()

    def __getitem__(self, idx):
        res = self.seq.__getitem__(idx)
        if len(res) == 1 and type(idx) is int:
            return res
        if type(res) is NaturalClass and type(idx) is int:
            return res
        return Sequence(res)

    def __setitem__(self, idx, val):
        self.seq[idx] = val

    def __iter__(self):
        return self.seq.__iter__()

    def __hash__(self):
        return hash(self.__str__())

    def __iadd__(self, other):
        if type(other) is str:
            self.seq += [other]
        elif type(other) is Segment:
            self.seq += [other]
        else:
            assert(type(other) is Sequence)
            self.seq += list(other.seq)
        return self

    def __add__(self, other):
        if type(other) is str:
            return self.seq + [other]
        elif type(other) is Segment:
            return self.seq + [other.ipa]
        assert(type(other) is Sequence)
        return self.seq + list(other.seq)
