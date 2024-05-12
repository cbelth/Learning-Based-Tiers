import numpy as np
import random
import os
from collections import defaultdict
import argparse

import sys
sys.path.append('../src/')

from utils import load
from d2l import D2L
from plp_grammar import PLP_Grammar
from gg_baseline import GGbaseline

class Exp:
    '''
    A class for running natural language experiments from the paper (Turkish, Finnish, Latin).
    '''
    def __init__(self, path, res_path, name, model_builder):
        self.path = path
        self.res_path = res_path
        self.name = name
        self.model_builder = model_builder
        self.accs = list()
        self.is_ours = type(model_builder()) is D2L or type(model_builder()) is PLP_Grammar
        self.is_gg = type(model_builder()) is GGbaseline
        self.multi_rules = self.is_ours and type(model_builder()) is PLP_Grammar
        if self.is_ours:
            self.rules = defaultdict(int)

    def run(self, frac_train=0.8, num_train=None, num_seeds=30, overwrite=True):
        if frac_train and num_train:
            raise ValueError(f'Cannot have both :frac_train: and :num_train: valued (one should be None)')

        abs_path = os.path.dirname(__file__)
        self.res_path = f'{abs_path}/{self.res_path}/{self.name}'
        if overwrite and os.path.exists(self.res_path):
            os.system(f'rm -r {self.res_path}')
        if not os.path.exists(self.res_path):
            os.mkdir(self.res_path)
        
        pairs, freqs = load(self.path, skip_header=True)
        num_train = num_train if num_train else int(round(len(pairs) * frac_train))
        
        for seed in range(num_seeds):
            if not overwrite and os.path.exists(f'{self.res_path}/{seed}_acc.txt'):
                continue
            print(f'Running seed {seed}')
            random.seed(seed)
            train = set()
            while len(train) < num_train:
                train.add(tuple(random.choices(pairs, weights=freqs, k=1))[0])
            test = list()
            for pair in pairs:
                if pair not in train:
                    test.append(pair)
            model = self.model_builder()
            model.train(train)
            acc, errs = model.accuracy(test, return_errors=True)
            self.write(seed, acc, errs)
            self.accs.append(acc)
            if self.is_ours:
                self.rules[(model.rule if not self.multi_rules else tuple(model.rules.items()))] += 1
            if self.is_gg:
                with open(f'{self.res_path}/{seed}_proj.txt', 'w') as f:
                    for proj, con, str_con in model.proj_info:
                        f.write(f'{proj}\t{con}\t{str_con}\n')
        if self.is_ours:
            self.rules = sorted(self.rules.items(), reverse=True, key=lambda it: it[-1])
        return self
    
    def write(self, seed, acc, errs):
        with open(f'{self.res_path}/{seed}_acc.txt', 'w') as f:
            f.write(f'{acc}\n')
        with open(f'{self.res_path}/{seed}_errs.txt', 'w') as f:
            f.write('uf\tpred\tsf\n')
            for uf, pred, sf in errs:
                f.write(f'{uf}\t{pred}\t{sf}\n')
    
    def print_res(self):
        m = format(np.mean(self.accs), '0.4f')

        v = format(np.std(self.accs), '0.2f')
        print(f'${m} \pm {v}$')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp-name', '-e', type=str, required=True, help='Turkish-CHILDES|Turkish-Morpho|Finnish|Latin')
    parser.add_argument('--model', '-m', type=str, required=True, help='D2L|GR|trigram|TSLIA|GG|LSTM')
    args = parser.parse_args()

    if args.exp_name == 'Turkish-CHILDES':
        ipa_file = '../data/turkish/ipa.txt'
        path = '../data/turkish/childes.txt'
        res_path = '../results/turkish/childes/'
    elif args.exp_name == 'Turkish-Morpho':
        ipa_file = '../data/turkish/ipa.txt'
        path = '../data/turkish/morpho.txt'
        res_path = '../results/turkish/morpho/'
    elif args.exp_name == 'Finnish':
        ipa_file = '../data/finnish/ipa.txt'
        path = '../data/finnish/finnish.txt'
        res_path = '../results/finnish/'
    elif args.exp_name == 'Latin':
        ipa_file = '../data/latin/ipa.txt'
        path = '../data/latin/latin.txt'
        res_path = '../results/latin/'
    else:
        raise ValueError('Experiment name must be one of Turkish-CHILDES|Turkish-Morpho|Finnish|Latin')

    if args.model == 'D2L':
        model_builder = lambda: D2L(ipa_file=ipa_file, verbose=False) if 'Turkish' not in args.exp_name else lambda: PLP_Grammar(ipa_file=ipa_file, verbose=False)
    elif args.model == 'trigram':
        from ngram import NGram
        model_builder = lambda: NGram(ipa_file=ipa_file, verbose=False)
    elif args.model == 'GR':
        from gr_baseline import GRbaseline
        model_builder = lambda: GRbaseline(ipa_file=ipa_file, verbose=False)
    elif args.model == 'GG':
        model_builder = lambda: GGbaseline(ipa_file=ipa_file, verbose=False)
    elif args.model == 'TSLIA':
        from tslia import TSLIA
        model_builder = lambda: TSLIA(ipa_file=ipa_file, verbose=False)
    elif args.model == 'LSTM':
        from lstm_baseline import LSTMbaseline
        model_builder = lambda: LSTMbaseline(ipa_file=ipa_file, verbose=False)
    else:
        raise ValueError('Model must be one of D2L|GR|trigram|TSLIA|GG|LSTM')
        
    model = Exp(
        path=path,
        res_path=res_path,
        name=args.model,
        model_builder=model_builder).run(
            num_train=1000 if 'Turkish' in args.exp_name else None,
            frac_train=0.8 if 'Turkish' not in args.exp_name else None,
            num_seeds=30)