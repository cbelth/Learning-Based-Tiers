from hmmlearn import hmm
import numpy as np
from collections import defaultdict

from phonotactic_model import PhonotacticModel
from tier import Tier

class GRbaseline(PhonotacticModel):
    def __init__(self, 
                 num_hidden_states=2,
                 ipa_file='../data/ipa.txt',
                 underspec=True,
                 verbose=True):

        super().__init__(ipa_file=ipa_file, add_segs=True, underspec=underspec, verbose=verbose)

        self.hmm = hmm.MultinomialHMM(n_components=num_hidden_states, n_iter=100)

    def train(self, pairs):
        _ = self.get_sfs(pairs)
        seqs, lens = self.get_seqs(pairs)
        self.hmm.fit(seqs, lens) # fit HMM
        self.hidden = self.get_hidden(pairs)
        self.compute_tier(pairs)
        self.compute_stats(pairs)

        return self

    def score(self, candidates):
        return list(self.boltzmann_score(cand) for cand in candidates)

    def compute_unigram(self, pairs):
        unigrams = defaultdict(int)
        for _, sf in pairs:
            for seg in sf:
                unigrams[seg] += 1
        self.unigrams = dict(unigrams)

    def compute_bigram(self, pairs):
        bigrams = defaultdict(int)
        for _, sf in pairs:
            for i in range(1, len(sf)):
                bg = sf[i-1:i+1]
                bigrams[bg] += 1
        self.bigrams = dict(bigrams)

    def compute_tier_unigram(self, pairs):
        unigrams = defaultdict(int)
        for _, sf in pairs:
            tier = self.tier.project(sf)
            for seg in tier:
                unigrams[seg] += 1
        self.tier_unigrams = dict(unigrams)

    def compute_tier_bigram(self, pairs):
        bigrams = defaultdict(int)
        for _, sf in pairs:
            tier = self.tier.project(sf)
            for i in range(1, len(tier)):
                bg = tier[i-1:i+1]
                bigrams[bg] += 1
        self.tier_bigrams = dict(bigrams)

    def compute_tier(self, pairs):
        alternating = set()
        for uf, sf in pairs:
            for i in range(len(sf)):
                uf_seg, sf_seg = uf[i], sf[i]
                if uf_seg != sf_seg:
                    alternating.add(uf_seg)
                    alternating.add(sf_seg)
        # set the delset as the hidden state that contains the fewest alternating segs (hopefully zero)
        delset = sorted(self.hidden, reverse=False, key=lambda h: len(alternating.intersection(h)))[0]
        tierset = self.seginv.complement(delset)
        self.tier = Tier(tierset=tierset, seginv=self.seginv)

    def get_seqs(self, pairs):
        alph = set()
        for _, sf in pairs:
            alph.update(sf)
        self.alph = dict((seg, i) for i, seg in enumerate(sorted(alph)))
        seqs = list()
        lens = list()
        for _, sf in pairs:
            seq = list()
            for seg in sf:
                seq.append([self.alph[seg]])
            seqs.append(seq)
            lens.append(len(seq))
        return np.concatenate(seqs), lens

    def get_hidden(self, pairs):
        '''
        Compute the hidden states of the HMM, which correspond to possible tiers.
        '''
        seg_to_states = defaultdict(list)
        for _, seq in pairs:
            dec = self.hmm.decode([list(self.alph[seg] for seg in seq)])[1]
            for seg, d in zip(seq, dec):
                seg_to_states[seg].append(d)
                
        hidden = defaultdict(set)
        for seg, states in seg_to_states.items():
            h = max(set(states), key=states.count)
            hidden[h].add(seg)
        return tuple(sorted((tuple(sorted(h)) for h in hidden.values())))

    def plog(self, x, counts_dict, num_segs):
        Z = sum(counts_dict.values())
        count = counts_dict[x] + 0.5 if x in counts_dict else 0.5
        return np.log(count / (Z + num_segs * 0.5))

    def pw_MI(self, bg, counts_dict, num_segs, unigrams_dict):
        return  -self.plog(bg, counts_dict, num_segs) + self.plog(bg[0], unigrams_dict, num_segs) + self.plog(bg[1], unigrams_dict, num_segs)

    def boltzmann_score(self, seq):
        '''
        Eq. (15) on pg. 882 of Goldsmith & Riggle
        '''
        score = 0
        for i in range(1, len(seq)):
            score += self.plog(seq[i], self.unigrams, self.num_segs) # U(w)
            bg = seq[i-1:i+1]
            score -= self.pw_MI(bg, self.bigrams, self.num_segs, self.unigrams) # M_1(w)
        tier = self.tier.project(seq)
        for i in range(1, len(tier)):
            bg = tier[i-1:i+1]
            score -= self.pw_MI(bg, self.tier_bigrams, self.num_tier_segs, self.tier_unigrams) # M_2(w)
        return score

    def compute_stats(self, pairs):
        self.compute_unigram(pairs)
        self.compute_bigram(pairs)
        self.compute_tier_unigram(pairs)
        self.compute_tier_bigram(pairs)

        self.num_segs = len(self.seginv.segments)
        self.num_tier_segs = len(self.tier.tierset)
