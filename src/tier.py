from sequence import Sequence

class Tier:
    '''
    Tier object.
    '''
    def __init__(self, tierset, seginv):
        self.tierset = tierset
        self.seginv = seginv
        self.delset = self.seginv.complement(tierset) if type(tierset) is set else self.seginv.extension_complement(tierset)

    def project(self, uf, sf=None):
        uf_tier = Sequence('', self.seginv)
        if sf:
            sf_tier = Sequence('', self.seginv)
        for i in range(len(uf)):
            seg = uf[i]
            if seg in self.tierset:
                uf_tier += seg
                if sf:
                    sf_tier += sf[i]
        if sf:
            return uf_tier, sf_tier
        return uf_tier

    def __str__(self):
        return f'{self.tierset}' if type(self.tierset) is not set else  f'{tuple(sorted(self.tierset))}'

    def __repr__(self):
        return self.__str__()

    def __contains__(self, item):
        return item in self.tierset