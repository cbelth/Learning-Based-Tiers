import unittest
import sys
import random
sys.path.append('../src/')
from utils import load
from d2l import D2L
from plp_grammar import PLP_Grammar
from discrepancy import Discrepancy

class TestD2L(unittest.TestCase):
    def test_init(self):
        model = D2L()

    def check(self, model, data):
        for uf, sf in data:
            pred = model.produce(uf)
            if pred != sf:
                print(f'uf: {uf}, sf: {sf}, pred: {pred}')
            assert(pred == sf)
    
    def test_paper_ex(self):
        train = [
            ('ʃokuSiS', 'ʃokuʃiʃ'), 
            ('apʃaS', 'apʃaʃ'),
            ('ʃuniS', 'ʃuniʃ'),
            ('sokiS', 'sokis'),
            ('sigoSiS', 'sigosis'),
            ('utS', 'uts')
        ]
        model = D2L(ipa_file='../data/finley/ipa.txt', verbose=False)
        model.train(train)
        self.check(model, train)
        assert(f'{model.rule}' == "{S} --> ('ant',) / {+strid} __ / {+strid}")

    def test_turkish(self):
        pairs, _ = load('../data/turkish/childes.txt', skip_header=True)
        model = PLP_Grammar(ipa_file='../data/turkish/ipa.txt', verbose=False)
        model.train(pairs)

        assert(len(model.rules) == 2)
        harmony_discrep = Discrepancy(('back', 'round'))
        voi_discrep = Discrepancy(('voice',))
        assert(harmony_discrep in model.rules)
        assert(voi_discrep in model.rules)
        
        harmony_rule = model.rules[harmony_discrep]
        voi_rule = model.rules[voi_discrep]
        assert(f'{harmony_rule.tier.tierset}' == '{-cons}')
        assert(f'{voi_rule.tier.tierset}' == '{*}')

    def test_finnish(self):
        pairs, _ = load('../data/finnish/finnish.txt', skip_header=True)
        model = D2L(ipa_file='../data/finnish/ipa.txt', verbose=False)
        model.train(pairs)

        non_neut_vowels = model.seginv.extension({'-cons'}).difference({'i', 'e'})
        assert(model.rule.tier.tierset == non_neut_vowels)
        assert(model.rule.lc == non_neut_vowels)
        assert(model.default.vals == ('-',))

    def test_latin(self):
        pairs, _ = load('../data/latin/latin.txt', skip_header=True)
        model = D2L(ipa_file='../data/latin/ipa.txt', verbose=False)
        model.train(pairs)

        assert(f'{model.rule}' == "{L} <-- ('lat',) / {+cons} __ / {+cons}")

if __name__ == "__main__":
    unittest.main()
