import torch
from pytorch_lightning import LightningModule
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

class LSTM(LightningModule):
    def __init__(
        self,
        dm,
        input_dim: int,
        output_dim: int,
        emb_dim: int = 128,
        hid_dim: int = 256,
        dropout: float = 0.1,
        num_rnn_layers: int = 1,
        learning_rate: float = 0.001,
        adam_epsilon: float = 1e-8,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        train_batch_size: int = 32,
        eval_batch_size: int = 32
    ):
        super().__init__()
        
        self.save_hyperparameters()

        self.dm = dm
        self.pad_id = self.dm.stoi['[PAD]']
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_id)

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(input_size=emb_dim, 
                           hidden_size=hid_dim, 
                           num_layers=num_rnn_layers, 
                           bidirectional=True, 
                           batch_first=True,
                           dropout=0.0 if num_rnn_layers == 1 else dropout)
        self.fc = nn.Linear(hid_dim * 2, output_dim)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer
        
    def forward(self, x):
        if type(x) is str:
            x = torch.tensor(self.dm._tokenize(x)).unsqueeze(0).to('cuda:0')
        emb = self.embedding(x)
        output, (hidden, cell) = self.rnn(emb)
        p = F.softmax(self.fc(output), dim=2)
        return p

    def _compute_loss(self, batch, batch_idx):
        x = batch['src']
        y = batch['tgt']
        
        pad_fill = torch.full(y.shape, self.pad_id).to('cuda:0')
        masked_y = torch.where(y != x, y, pad_fill) # Take y_i when y_i != x_i, otherwise self.pad_id, which self.criterion should ignore

        y_hat = self(x)
    
        loss = self.criterion(y_hat.view(-1, self.output_dim), masked_y.view(-1))
        return loss

    def training_step(self, batch, batch_idx):
        return self._compute_loss(batch, batch_idx)
    
    def training_epoch_end(self, train_step_outputs):
        total_loss = sum(loss['loss'].item() for loss in train_step_outputs)
        self.print('train_loss', total_loss / len(train_step_outputs))
    
    def validation_step(self, batch, batch_idx):
        return self._compute_loss(batch, batch_idx)
    
    def validation_epoch_end(self, val_step_outputs):
        total_loss = sum(loss.item() for loss in val_step_outputs)
        self.print('valid_loss', total_loss / len(val_step_outputs))
        self.log('val_loss', total_loss / len(val_step_outputs))