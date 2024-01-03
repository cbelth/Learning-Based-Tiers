import os
import random
import numpy as np
from scipy.stats import ttest_ind

from utils import load
from exp.exp import Exp

class McMullin(Exp):
    def acc(self, items, model, assim=True):
        if not self.is_gg:
            c, n = 0, 0
            for uf in items:
                stem = uf[:-2]
                suf = uf[-2:]
                assert('ɹ' not in stem)
                n += 1
                l_opt = stem.replace('L', 'l') + suf
                r_opt = stem.replace('L', 'ɹ') + suf
                if assim:
                    harmonizing = l_opt if 'l' in suf else r_opt
                else:
                    harmonizing = r_opt if 'l' in suf else l_opt
                choice = model.choose(uf, opt1=l_opt, opt2=r_opt)

                if choice == harmonizing:
                    c += 1
            return c / n
        else:
            l_opts = list()
            r_opts = list()
            for uf in items:
                stem = uf[:-2]
                suf = uf[-2:]
                assert('ɹ' not in stem)
                l_opts.append(stem.replace('L', 'l') + suf)
                r_opts.append(stem.replace('L', 'ɹ') + suf)
            choices = model.batch_choose(opt1s=l_opts, opt2s=r_opts)
            c, n = 0, 0
            for uf, choice in zip(items, choices):
                stem = uf[:-2]
                suf = uf[-2:]
                n += 1
                l_opt = stem.replace('L', 'l') + suf
                r_opt = stem.replace('L', 'ɹ') + suf
                if assim:
                    harmonizing = l_opt if 'l' in suf else r_opt
                else:
                    harmonizing = r_opt if 'l' in suf else l_opt

                if choice == harmonizing:
                    c += 1
            return c / n         

    def run(self, rand=None, assim=True, num_seeds=30):
        
        self.a1s = list()
        self.a2s = list()
        self.a3s = list()

        if rand:
            abs_path = os.path.dirname(__file__)
            self.res_path = f'{abs_path}/../{self.res_path}/{self.name}'
            if os.path.exists(self.res_path):
                os.system(f'rm -r {self.res_path}')
            os.mkdir(self.res_path)

        train, _ = load(f'{self.path}-train.txt', skip_header=True)    
        
        test = list(line.strip() for line in open('../data/mcmullin/test.txt'))
        short = list(filter(lambda w: w.index('L') == 4, test))
        medium = list(filter(lambda w: w.index('L') == 2, test))
        long = list(filter(lambda w: w.index('L') == 0, test))

        for seed in range(num_seeds):
            random.seed(seed)
            _train = list()
            for i in range(1): # repeated n times
                _tr = list(train)
                random.shuffle(_tr)
                _train.extend(_tr)
            model = self.model_builder()
            model.train(_train)

            self.a1s.append(self.acc(short, model, assim=assim))
            self.a2s.append(self.acc(medium, model, assim=assim))
            self.a3s.append(self.acc(long, model, assim=assim))

            if self.is_ours:
                self.rules[model.rule] += 1
            if self.is_gg:
                with open(f'{self.res_path}/{seed}_proj.txt', 'w') as f:
                    for proj, con, str_con in model.proj_info:
                        f.write(f'{proj}\t{con}\t{str_con}\n')

        if rand:
            with open(f'{self.res_path}/acc_train.txt', 'w') as f:
                for acc in self.a1s:
                    f.write(f'{acc}\n')
            with open(f'{self.res_path}/acc_train_like.txt', 'w') as f:
                for acc in self.a2s:
                    f.write(f'{acc}\n')
            with open(f'{self.res_path}/acc_novel.txt', 'w') as f:
                for acc in self.a3s:
                    f.write(f'{acc}\n')

            gen_acc1, gen_acc2, gen_acc3 = self.compare(rand)
            with open(f'{self.res_path}/ttests.txt', 'w') as f:
                f.write(f'{gen_acc1},{gen_acc2},{gen_acc3}\n')

        if self.is_ours:
            self.rules = sorted(self.rules.items(), reverse=True, key=lambda it: it[-1])

        return self
    
    def compare(self, rand):
        p1 = ttest_ind(rand.a1s, self.a1s, equal_var=False, alternative='less').pvalue
        p2 = ttest_ind(rand.a2s, self.a2s, equal_var=False, alternative='less').pvalue
        p3 = ttest_ind(rand.a3s, self.a3s, equal_var=False, alternative='less').pvalue
        
        g1 = np.mean(self.a1s) >= np.mean(rand.a1s)
        g2 = np.mean(self.a2s) >= np.mean(rand.a2s)
        g3 = np.mean(self.a3s) >= np.mean(rand.a3s)
        
        return g1 and p1 < 0.01, g2 and p2 < 0.01, g3 and p3 < 0.01

    def print_res(self):
        m1 = format(np.mean(self.a1s), '0.4f')
        m2 = format(np.mean(self.a2s), '0.4f')
        m3 = format(np.mean(self.a3s), '0.4f')
        
        v1 = format(np.std(self.a1s), '0.2f')
        v2 = format(np.std(self.a2s), '0.2f')
        v3 = format(np.std(self.a3s), '0.2f')
        print(f'${m1} \pm {v1}$')
        print(f'${m2} \pm {v2}$')
        print(f'${m3} \pm {v3}$')