from phonotactic_model import PhonotacticModel

import os
import glob
import random
import numpy as np

class GGbaseline(PhonotacticModel):
    def __init__(self, 
                 gain=50,
                 ncons=100,
                 ipa_file='../data/ipa.txt',
                 underspec=True,
                 verbose=True):

        super().__init__(ipa_file=ipa_file, add_segs=False, underspec=underspec, verbose=verbose)
        self.gain = gain
        self.ncons = ncons

        self.root_dir = os.path.dirname(os.path.abspath(__file__))
        self.path = f'{self.root_dir}/gg/data/temp'
        self.test_path = f'{self.path}/TestingData.txt'
        self._write_ipa(ipa_file)

    def score(self, candidates):
        '''
        Forms with more violations get lower values (Hayes & Wilson, 2008, p. 384)
        '''
        with open(f'{self.test_path}', 'w') as f:
            for cand in candidates:
                _cand = ' '.join(list(cand))
                f.write(f'{_cand}\n')

        self.test_script_path = self._write_script(name='test', cmd=f'python run_sim.py test {self.out_dir}/grammar.txt {self.test_path}')

        os.system(f'bash {self.test_script_path} >& {self.path}/test.out')

        _test_dirs = glob.glob(f'{self.out_dir}/test_*')
        if len(_test_dirs) > 0:
            tableau_dir = _test_dirs[0]
            tableau = f'{tableau_dir}/tableau.txt'
            cand_id = 0
            hx_id = 1
            scores = dict()
            if os.path.exists(tableau):
                with open(tableau, 'r') as f:
                    next(f)
                    for line in f:
                        line = line.strip().split('\t')
                        cand = ''.join(line[cand_id].split())
                        hx = float(line[hx_id])
                        scores[cand] = hx
            else:
                print(f'Tableau does not exist at {tableau}. Using random scores instead.')
                scores = dict((cand, 0) for cand in candidates)
            os.system(f'rm -r {tableau_dir}/')
        else:
            print(f'No test directory at {self.out_dir}/test_*. Using random scores instead. Probably something crashed while running.')
            scores = dict((cand, 0) for cand in candidates)

        os.system(f'rm {self.test_path}')

        return list(scores[cand] for cand in candidates)

    def train(self, pairs):
        sfs = self.get_sfs(pairs)
        with open(f'{self.path}/LearningData.txt', 'w') as f:
            for sf in sfs:
                sf = ' '.join(list(sf))
                f.write(f'{sf}\n')

        if self.verbose:
            print(f'Running simulation. Track progress at {self.path}/train.out')
        script_path = self._write_script(name='run', cmd=f'python run_sim.py temp wb gain{self.gain} ncons{self.ncons}')
        os.system(f'bash {script_path} >& {self.path}/train.out')
        if self.verbose:
            print('Simulation complete.')

        out_dir = open(f'{self.path}/train.out', 'r').readlines()[-1].strip()
        os.system(f'rm -r {self.path}/out/') # clear old output
        os.system(f'mv {out_dir} {self.path}/out') # move results to `out/' dir
        if self.verbose:
            print(f'Results moved to {self.path}/out')
        projections = set()
        self.proj_info = list()
        if os.path.exists(f'{self.path}/out/projections/projections.txt'):
            with open(f'{self.path}/out/projections/projections.txt', 'r') as f:
                next(f)
                for line in f:
                    proj = line.strip().split('\t')[0]
                    if self.verbose:
                        print(proj)
                    projections.add(proj)
            if os.path.exists(f'{self.path}/out/output_final/grammar.txt'):
                with open(f'{self.path}/out/output_final/grammar.txt', 'r') as f:
                    next(f)
                    for line in f:
                        line = line.strip().split('\t')
                        proj = line[0]
                        if proj in projections:
                            con = line[1]
                            str_con = line[-1]
                            if self.verbose:
                                print(proj, con, str_con)
                            self.proj_info.append((proj, con, str_con))

                        
                
        out_dir = f'{self.path}/out' # update new out_dir to `out/`

        baseline_out_dir = f'{out_dir}/output_baseline'
        gg_out_dir = f'{out_dir}/output_final'

        self.out_dir = gg_out_dir if os.path.exists(gg_out_dir) else baseline_out_dir

        self.test_script_path = self._write_script(name='test', cmd=f'python run_sim.py test {self.out_dir}/grammar.txt {self.test_path}')

        return self

    def _write_ipa(self, ipa_file):
        with open(f'{self.path}/Features.txt', 'w') as wf:
            with open(ipa_file, 'r') as rf:
                for i, line in enumerate(rf):
                    if i == 0:
                        wf.write(f"{line.replace('SEG', '')}")
                    elif line.split('\t')[0].isupper():
                        continue
                    else:
                        wf.write(f"{line.replace('?', '0')}")

    def _write_script(self, name, cmd):
        script_path = f'{self.path}/{name}.sh'
        with open(script_path, 'w') as f:
            f.write('#!/bin/bash\n')
            f.write('\n')
            f.write(f'cd {self.root_dir}/gg/code/\n')
            f.write(f'{cmd}\n')

        os.system(f'chmod u+x {script_path}')

        return script_path

    def batch_choose(self, opt1s, opt2s):
        scores = self.score(opt1s + opt2s)
        score1s = scores[:len(opt1s)]
        score2s = scores[len(opt1s):]

        choices = list()
        for opt1, opt2, s1, s2 in zip(opt1s, opt2s, score1s, score2s):
            if s1 > s2:
                choices.append(opt1)
            elif s2 > s1:
                choices.append(opt2)
            else: # s1 == s2
                choices.append(random.choice([opt1, opt2]))
        return choices

    def accuracy(self, test, return_errors=False):
        all_candidates = set()
        for uf, _ in test:
            all_candidates.update(self._get_candidates(uf))
        all_candidates = list(all_candidates)
        scores = dict((cand, score) for cand, score in zip(all_candidates, self.score(all_candidates)))

        errors = list()
        t, c, = 0, 0
        for uf, sf in test:
            cands = self._get_candidates(uf)
            scored = list(scores[cand] for cand in cands)
            argmax = np.argmax(scored)
            pred = cands[argmax]
            if sf == pred:
                c += 1
            elif return_errors:
                errors.append((uf, pred, sf))
            t += 1
        if return_errors:
            return c / t if t > 0 else 0, errors
        return c / t if t > 0 else 0