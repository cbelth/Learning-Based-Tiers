LEFT_WORD_BOUNDARY = '⋊'
RIGHT_WORD_BOUNDARY = '⋉'
SYLLABLE_BOUNDARY = '.'
PRIMARY_STRESS = 'ˈ'
SECONDARY_STRESS = 'ˌ'
LONG = 'ː'
NASALIZED = '\u0303'
EMPTY_STRING = 'λ'
UNKNOWN_CHAR = '?'

from collections import defaultdict
import numpy as np

def load(fname, sep='\t', skip_header=False, seginv=False):
    pairs = list()
    freqs = list()
    with open(fname, 'r') as f:
        if skip_header:
            next(f)
        for line in f:
            line = line.strip().split(sep)
            line_length = len(line)
            if line_length == 2:
                uf, sf = line
                freq = 0
            if line_length == 3:
                uf, sf, freq = line
            elif line_length == 4:
                _, uf, sf, freq = line
            freq = float(freq)
            pairs.append((uf, sf))
            freqs.append(freq)
    if seginv:
        _alph = set()
        for uf, sf in pairs:
            for i in range(len(uf)):
                seg = uf[i]
                _alph.add(seg)
            for i in range(len(sf)):
                seg = sf[i]
                _alph.add(seg)
        return pairs, freqs, _alph
    return pairs, freqs

def tolerance_principle(n, c, e=None):
    if n == c:
        return True
    if e == None:
        e = n - c
    return c > 2 and c > n / 2 and e <= n / np.log(n)

def most_freq(l):
    item_to_count = defaultdict(int)
    for item in l:
        item_to_count[item] += 1
    argmax = None
    max_val = -1000
    for item, count in item_to_count.items():
        if count > max_val:
            argmax = item
            max_val = count
    return argmax