import numpy as np
import random
import os
from scipy.stats import ttest_ind

from utils import load
from exp.exp import Exp

class Finley(Exp):
    def acc(self, stems, model):
        c, n = 0, 0
        for stem in stems:
            uf = f'{stem}Su'
            n += 1
            su_opt, shu_opt = f'{stem}su', f'{stem}ʃu'
            harmonizing = su_opt if 's' in stem else shu_opt
            choice = model.choose(uf, opt1=su_opt, opt2=shu_opt)
            
            if choice == harmonizing:
                c += 1
        return c / n

    def run(self, rand=None, num_seeds=30, num_times=5):
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
        test_train = list(line.strip() for line in open(f'{self.path}-test-train.txt', 'r'))
        test_train_like = list(line.strip() for line in open(f'{self.path}-test-train-like.txt', 'r'))
        test_new = list(line.strip() for line in open(f'{self.path}-test-new.txt', 'r'))
        
        for seed in range(num_seeds):
            random.seed(seed)
            
            _train = list()
            for _ in range(num_times): # repeated n times
                _tr = list(train)
                random.shuffle(_tr)
                _train.extend(_tr)
            model = self.model_builder()
            model.train(_train)

            self.a1s.append(self.acc(test_train, model))
            self.a2s.append(self.acc(test_train_like, model))
            self.a3s.append(self.acc(test_new, model))
            
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

    def compare(self, rand):
        p1 = ttest_ind(rand.a1s, self.a1s, equal_var=False, alternative='less').pvalue
        p2 = ttest_ind(rand.a2s, self.a2s, equal_var=False, alternative='less').pvalue
        p3 = ttest_ind(rand.a3s, self.a3s, equal_var=False, alternative='less').pvalue
        
        g1 = np.mean(self.a1s) >= np.mean(rand.a1s)
        g2 = np.mean(self.a2s) >= np.mean(rand.a2s)
        g3 = np.mean(self.a3s) >= np.mean(rand.a3s)
        
        return g1 and p1 < 0.01, g2 and p2 < 0.01, g3 and p3 < 0.01