import torch
from pytorch_lightning import Trainer
import optuna as opt
import numpy as np

from model import Model
from nn.data_module import DataModule
from nn.lstm import LSTM

class LSTMbaseline(Model):
    def __init__(self, 
                 ipa_file='../data/ipa.txt',
                 underspec=True,
                 verbose=True):

        super().__init__(ipa_file=ipa_file, add_segs=False, underspec=underspec, verbose=verbose)
        self.dm = DataModule()
                
    def train(self, pairs):
        self.dm.prepare(pairs)
        def optim_objective(trial):
            losses = list()
            for i in range(1):
                lr = trial.suggest_float('lr', 0.0001, 0.1)
                max_epochs = trial.suggest_int('max_epochs', 1, 10)
                emb_dim = trial.suggest_categorical('emb_dim', [16, 32, 64, 128, 256, 512])
                hid_dim = trial.suggest_categorical('hid_dim', [16, 32, 64, 128, 256, 512])

                dm = DataModule()
                dm.prepare(pairs)
                dm.kfold_split()
                lstm = LSTM(input_dim=dm.alph_size, 
                            output_dim=dm.alph_size, 
                            emb_dim=emb_dim,
                            hid_dim=hid_dim,
                            learning_rate=lr,
                            dm=dm)
                lstm.train()
                trainer = Trainer(max_epochs=max_epochs, gpus=1, val_check_interval=1.0)
                trainer.fit(lstm, datamodule=dm)
                loss = trainer.callback_metrics['val_loss'].item()
                losses.append(loss)
            return np.mean(losses)
        
        study = opt.create_study(direction='minimize')
        study.optimize(optim_objective, n_trials=30)

        best = study.best_trial
        print("  Value: {}".format(best.value))

        print("  Params: ")
        for key, value in best.params.items():
            print("    {}: {}".format(key, value))
        print('Running on full dataset...')
            
        lr = best.params['lr']
        emb_dim = best.params['emb_dim']
        hid_dim = best.params['hid_dim']
        max_epochs = best.params['max_epochs']

        self.dm.all_train_split()
        self.lstm = LSTM(input_dim=self.dm.alph_size, 
                         output_dim=self.dm.alph_size, 
                         emb_dim=emb_dim,
                         hid_dim=hid_dim,
                         learning_rate=lr,
                         dm=self.dm)
        self.lstm.train()
        trainer = Trainer(max_epochs=max_epochs, gpus=1, val_check_interval=1.0)
        trainer.fit(self.lstm, datamodule=self.dm)

        self.lstm.eval()
        self.lstm.to('cuda:0')

    def produce(self, uf):
        _pred = self.dm._detokenize(self.lstm(uf).argmax(dim=2).squeeze(0).tolist()[1:-1])

        pred = ''
        for i in range(len(uf)):
            if uf[i].islower():
                pred += uf[i]
            else:
                pred += _pred[i]
        
        return pred

    def choose(self, uf, opt1, opt2):
        pred = self.produce(uf)
        if pred == opt1:
            return pred
        if pred == opt2:
            return pred
        return random.choice([opt1, opt2])