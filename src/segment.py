class Segment:
    '''
    Class representing a Segment as a vector of features, which simultaneously behaves like an IPA str.
    '''
    def __init__(self, ipa, feature_vec=[]):
        self.ipa = ipa
        self.feature_vec = feature_vec

    def __eq__(self, other):
        return f'{self}' == f'{other}'

    def __neq__(self, other):
        return not self.__eq__(other)

    def __gt__(self, other):
        return self.ipa.__gt__(other if type(other) is str else other.ipa)

    def __ge__(self, other):
        return self.ipa.__ge__(other if type(other) is str else other.ipa)

    def __lt__(self, other):
        return self.ipa.__lt__(other if type(other) is str else other.ipa)

    def __le__(self, other):
        return self.ipa.__le__(other if type(other) is str else other.ipa)

    def __hash__(self):
        return hash(self.ipa)

    def __str__(self):
        return self.ipa

    def __repr__(self):
        return self.__str__()

    # overload string functions
    def __len__(self):
        return 1

    def __iter__(self):
        return self.ipa[0].__iter__()

    def count(self, val):
        return self.ipa.count(val)

    def __getitem__(self, idx):
        return self.ipa[idx]