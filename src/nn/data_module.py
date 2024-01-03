import random
import torch
from sklearn.utils import gen_batches
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, TensorDataset, DataLoader

class DataModule(LightningDataModule):
    def __init__(
        self,
        train_batch_size: int = 32,
        val_batch_size: int = 32
    ):
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        super().__init__()

    def _tokenize(self, seq, max_len=None):
        if max_len is None:
            max_len = len(seq)
        padding = [self.stoi['[PAD]']] * (max_len - len(seq))
        return [self.stoi['[SOS]']] + list(self.stoi[seg] if seg in self.stoi else 4 for seg in seq) + [self.stoi['[EOS]']] + padding

    def _detokenize(self, seq):
        return list(self.itos[seg] for seg in seq)

    def setup(self, stage: str):
        pass

    def prepare(self, pairs):
        max_len = 0
        self.stoi = {'[SOS]': 0, '[EOS]': 1, '[MASK]': 2, '[PAD]': 3, '[UNKNOWN]': 4}
        alph_idx = 5
        for uf, sf in pairs:
            for seg in uf + sf:
                if seg not in self.stoi:
                    self.stoi[seg] = alph_idx
                    alph_idx += 1
                if len(uf) > max_len:
                    max_len = len(uf)
        self.itos = dict((idx, seg) for seg, idx in self.stoi.items())
        self.alph_size = len(self.itos)
        
        self.dataset = list()
        for uf, sf in pairs:
            self.dataset.append((self._tokenize(uf, max_len=max_len), self._tokenize(sf, max_len=max_len)))

    def kfold_split(self):
        random.shuffle(self.dataset)

        num_train = int(0.8 * len(self.dataset))
        train = self.dataset[:num_train]
        val = self.dataset[num_train:]

        self.train = CustomDataset(list({'src': torch.tensor(it[0]).to('cuda:0'), 'tgt': torch.tensor(it[1]).to('cuda:0')} for it in train))
        self.val = CustomDataset(list({'src': torch.tensor(it[0]).to('cuda:0'), 'tgt': torch.tensor(it[1]).to('cuda:0')} for it in val))

    def all_train_split(self):
        random.shuffle(self.dataset)
        train = self.dataset
        val = self.dataset
        self.train = CustomDataset(list({'src': torch.tensor(it[0]), 'tgt': torch.tensor(it[1])} for it in train))
        self.val = CustomDataset(list({'src': torch.tensor(it[0]), 'tgt': torch.tensor(it[1])} for it in val))

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.train_batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.val_batch_size)

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.__getitem__(idx)

