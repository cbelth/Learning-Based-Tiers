class Discrepancy:
    def __init__(self, feat_diff):
        self.alternations = set()
        self.annotated = list()
        self.feat_diff = feat_diff

    def add(self, uf, i, uf_seg, sf_seg):
        self.annotated.append((uf, i, sf_seg))
        self.alternations.add((uf_seg, sf_seg))

    def get_alternating(self):
        return set(it[0] for it in self.alternations).union(it[1] for it in self.alternations)

    def get_alternating_ufs(self):
        return set(it[0] for it in self.alternations)

    def __contains__(self, item):
        return item in self.alternations

    def __str__(self):
        return self.feat_diff.__str__()

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.feat_diff == other.feat_diff

    def __neq__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.feat_diff)