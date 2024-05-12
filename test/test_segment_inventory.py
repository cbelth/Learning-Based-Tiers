from audioop import add
import unittest
import sys
sys.path.append('../src/')
from natural_class import NaturalClass
from segment_inventory import SegmentInventory

class TestSegmentInventory(unittest.TestCase):
    def test_ipa_files(self):
        alph = SegmentInventory('../data/ipa.txt', add_segs=True)
        assert(len(alph.check_unique()) == 0)
        alph = SegmentInventory('../data/finley/ipa.txt', add_segs=True)
        assert(len(alph.check_unique()) == 0)
        alph = SegmentInventory('../data/mcmullin/ipa.txt', add_segs=True)
        assert(len(alph.check_unique()) == 0)
        alph = SegmentInventory('../data/turkish/ipa.txt', add_segs=True)
        assert(alph.check_unique() == [{'ɰ', 'j'}]) # TODO maybe fix this
        alph = SegmentInventory('../data/finnish/ipa.txt', add_segs=True)
        assert(len(alph.check_unique()) == 0)
        alph = SegmentInventory('../data/latin/ipa.txt', add_segs=True)
        assert(len(alph.check_unique()) == 0)

    def test_getitem_1(self):
        seginv = SegmentInventory(add_segs=True)

        b = seginv['b']
        assert(seginv['b'].ipa == 'b')
        assert(seginv[','.join(b.feature_vec)].ipa == 'b')
        assert(seginv[b.feature_vec].ipa == 'b')
        assert(seginv[b] == b)

        p = seginv['p']
        assert(seginv['p'].ipa == 'p')
        assert(seginv[','.join(p.feature_vec)].ipa == 'p')
        assert(seginv[p.feature_vec].ipa == 'p')
        assert(seginv[p] == p)

    def test_getitem_2(self):
        seginv = SegmentInventory('../data/finley/ipa.txt', add_segs=True)
        S = seginv['S']
        assert(seginv['S'].ipa == 'S')
        assert(seginv[','.join(S.feature_vec)] == 'S')
        assert(seginv[S.feature_vec] == 'S')
        assert(seginv[S] == S)

    def test_feat_diff_1(self):
        seginv = SegmentInventory('../data/finley/ipa.txt', add_segs=True)
        assert(seginv.feat_diff('s', 'S') == {'ant'})
        assert(seginv.feat_diff('s', 'ʃ') == {'ant'})

    def test_extension_complement_1(self):
        seginv = SegmentInventory('../data/finley/ipa.txt', add_segs=True)
        nc = NaturalClass({'+strid'}, seginv)
        assert('s' in nc)
        assert('ʃ' in nc)
        assert('S' in nc)
        assert('p' not in nc)
        assert('a' not in nc)

        nc_comp = seginv.extension_complement(nc)
        assert('s' not in nc_comp)
        assert('ʃ' not in nc_comp)
        assert('S' not in nc_comp)
        assert('p' in nc_comp)
        assert('a' in nc_comp)

    def test_without_feats_1(self):
        seginv = SegmentInventory(add_segs=True)

        b = seginv['b']
        p = seginv['p']

        assert(seginv.without_feats('b', 'voice') == p)
        assert(seginv.without_feats(','.join(b.feature_vec), 'voice') == p)
        assert(seginv.without_feats(b.feature_vec, 'voice') == p)

    def test_with_feats_1(self):
        seginv = SegmentInventory(add_segs=True)

        b = seginv['b']
        p = seginv['p']

        assert(seginv.with_feats('p', 'voice') == b)
        assert(seginv.with_feats(','.join(p.feature_vec), 'voice') == b)
        assert(seginv.with_feats(p.feature_vec, 'voice') == b)

    def test_assimilate_1(self):
        alph = SegmentInventory('../data/turkish/ipa.txt', add_segs=True)

        assert(alph.assimilate('H', 'i', ('back', 'round')) == 'i')
        assert(alph.assimilate('H', 'ɯ', ('back', 'round')) == 'ɯ')
        assert(alph.assimilate('D', 'i', 'voice') == 'd')
        assert(alph.assimilate('k', 'ɯ', 'back') == None)

    def test_dissimilate_1(self):
        alph = SegmentInventory(add_segs=True)

        assert(alph.dissimilate('d', 'd', 'voice') == 't')
        assert(alph.dissimilate('g', 'd', 'voice') == 'k')
        assert(alph.dissimilate('m', 'd', 'voice') == None)

    def test_dissimilate_2(self):
        alph = SegmentInventory(ipa_file='../data/mcmullin/ipa.txt', add_segs=True)

        assert(alph.dissimilate('ɹ', 'ɹ', ('cor', 'ant', 'lat')) == 'l')
        assert(alph.dissimilate('l', 'l', ('cor', 'ant', 'lat')) == 'ɹ')

    def test_dissimilate_3(self):
        alph = SegmentInventory(ipa_file='../data/ipa.txt', add_segs=True)

        assert(alph.dissimilate('r', 'r', ('ant', 'lat')) == 'l')
        assert(alph.dissimilate('l', 'l', ('ant', 'lat')) == 'r')

if __name__ == "__main__":
    unittest.main()