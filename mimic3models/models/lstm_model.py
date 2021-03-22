import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Adam
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMClassifier(nn.Module):

    def __init__(self, tag_size, hidden_size,feat_size, emb_size,
                 bidirectional=False, dropout=0.2, aggregation_type='mean', depth=1):
        """
        :param hidden_size: size of recurrent cell
        :param pad_idx: id of the -PAD- token
        :param bidirectional: use rnn of bidirectional rnn         
        """
        super().__init__()
        self.bidirectional = bidirectional
        self.aggregation_type = aggregation_type
        #self.encoder = nn.Linear(feat_size, emb_size)
        # Create a (bidirectional) LSTM to encode sequence
        if depth > 1:
             self.lstm = nn.LSTM(feat_size, hidden_size, batch_first=True, num_layers=depth,
                                    dropout=dropout,bidirectional=bidirectional)
        else:
            self.lstm = nn.LSTM(feat_size, hidden_size, batch_first=True, 
                            bidirectional=bidirectional)
        
        # The output of the LSTM doubles if we use a bidirectional encoder.
        # we have the forward and backward hidden states 
        encoding_size = hidden_size * 2 if bidirectional else hidden_size
        self.combination_layer = nn.Linear(encoding_size, encoding_size)
        
        # Create affine layer to project to the classes 
        self.projection = nn.Linear(encoding_size, tag_size)
        # dropout layer for regularizetion of a sequence
        self.dropout_layer = nn.Dropout(p=dropout)
        #self.softplus = nn.Softplus()
        self.relu = nn.ReLU()
        #self.m = nn.LayerNorm(encoding_size)

    def forward(self, x, seq_mask=None, seq_len=None):
        # Encode the sentence using the LSTM.
        # this first step process the sequecnes in an optimized way
        # [B, M, hid_size]
        #e = self.encoder(x)
        #e = self.dropout_layer(e)
         
        outputs, (final, _) = self.lstm(x)

        if self.aggregation_type == 'last_state':
            #print(h.size())
            if self.bidirectional:
                h_T_fwd = final[0]
                h_T_bwd = final[1]
                h = torch.cat([h_T_fwd, h_T_bwd], dim=-1)
            else:
                h = final[-1]
            h = self.relu(self.combination_layer(h))
            h = self.dropout_layer(h)
        elif self.aggregation_type == 'mean':
            outputs = self.dropout_layer(outputs)
            h = self.relu(self.combination_layer(outputs))
            h = self.dropout_layer(h)
            #h = outputs
            h = h.mean(dim=1)
        elif self.aggregation_type == 'sum':
            outputs = self.dropout_layer(outputs)
            #outputs = outputs.mean(dim=1)
            h = self.relu(self.combination_layer(outputs))
            h = self.dropout_layer(h)
            #h = outputs
            h = h.sum(dim=1)
        
        # [B, num_class]
        # logits
        #h = self.m(h)
        logits = self.projection(h)
        
        return logits

class ComboClassifier(nn.Module):
    def __init__(self, feat_encoder, text_encoder, tag_size, hidden_size,feat_size, emb_size,
            bidirectional=False, dropout=0.2, aggregation_type='mean'):
        super().__init()
        self.feat_encoder = feat_encoder
        self.text_encoder = text_encoder
        self.projection = nn.Linear(encoding_size, tag_size)

    def forward(self, x, seq_mask=None, seq_len=None):
        pass



class RNNClassifier(nn.Module):

    def __init__(self, tag_size, hidden_size,feat_size, emb_size,
                 bidirectional=False, dropout=0.2, aggregation_type='mean'):
        """
        :param hidden_size: size of recurrent cell
        :param pad_idx: id of the -PAD- token
        :param bidirectional: use rnn of bidirectional rnn         
        """
        super().__init__()
        self.bidirectional = bidirectional
        self.aggregation_type = aggregation_type
        # Create a (bidirectional) LSTM to encode sequence
        self.rnn = nn.LSTM(emb_size, hidden_size, batch_first=True, 
                            bidirectional=bidirectional)
        
        # The output of the LSTM doubles if we use a bidirectional encoder.
        # we have the forward and backward hidden states 
        encoding_size = hidden_size * 2 if bidirectional else hidden_size
        self.combination_layer = nn.Linear(encoding_size, encoding_size)
        
        # Create affine layer to project to the classes 
        self.projection = nn.Linear(encoding_size, tag_size)
        # dropout layer for regularizetion of a sequence
        self.dropout_layer = nn.Dropout(p=dropout)
        self.relu = nn.Relu()

    def forward(self, x, seq_mask=None, seq_len=None, logits=None):
        # Encode the sentence using the LSTM.
        # this first step process the sequecnes in an optimized way
        # [B, M, hid_size]
         
        outputs, (final, _) = self.rnn(x)

        if self.aggregation_type == 'last_state':
            #print(h.size())
            if self.bidirectional:
                h_T_fwd = final[0]
                h_T_bwd = final[1]
                h = torch.cat([h_T_fwd, h_T_bwd], dim=-1)
            else:
                h = final[-1]
            h = self.relu(self.combination_layer(h))
            h = self.dropout_layer(h)
        elif self.aggregation_type == 'mean':
            outputs = self.dropout_layer(outputs)
            h = self.relu(self.combination_layer(outputs))
            h = self.dropout_layer(h)
            h = h.mean(dim=1)
        elif self.aggregation_type == 'sum':
            outputs = self.dropout_layer(outputs)
            h = self.relu(self.combination_layer(outputs))
            h = self.dropout_layer(h)
            h = outputs.sum(dim=1)
        
        # [B, num_class]
        # logits
        if logits:
            logits = self.projection(h)
        
            return logits
        else:
            return h

class LSTMSeqClassifier(nn.Module):

    def __init__(self, tag_size, hidden_size,feat_size, 
                 bidirectional=False, dropout=0.2):
        """
        :param hidden_size: size of recurrent cell
        :param pad_idx: id of the -PAD- token
        :param bidirectional: use rnn of bidirectional rnn         
        """
        super().__init__()
        self.bidirectional = bidirectional
        
        # Create a (bidirectional) LSTM to encode sequence
        self.lstm = nn.LSTM(feat_size, hidden_size, batch_first=True, 
                            bidirectional=bidirectional)
        
        # The output of the LSTM doubles if we use a bidirectional encoder.
        # we have the forward and backward hidden states 
        encoding_size = hidden_size * 2 if bidirectional else hidden_size
        self.combination_layer = nn.Linear(encoding_size, encoding_size)
        
        # Create affine layer to project to the classes 
        self.projection = nn.Linear(encoding_size, tag_size)
        # dropout layer for regularizetion of a sequence
        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, x, seq_mask=None, seq_len=None):
        # Encode the sentence using the LSTM.
        # this first step process the sequecnes in an optimized way
        # actual processing of the LSTM
        # [B, M, hid_size]
        outputs, _ = self.lstm(x)
        outputs = self.dropout_layer(outputs)
        # [B, M, hid_size]
        h = torch.relu(self.combination_layer(outputs))
        h = self.dropout_layer(h)
        
        # [B, M, num_class]
        # logits
        logits = self.projection(h)
        
        return logits

class LSTMFeatExtractor(nn.Module):
    def __init__(self, hidden_size,feat_size, emb_size,
                 bidirectional=False, dropout=0.2, aggregation_type='mean', depth=1):
        """
        :param hidden_size: size of recurrent cell
        :param pad_idx: id of the -PAD- token
        :param bidirectional: use rnn of bidirectional rnn         
        """
        super().__init__()
        self.bidirectional = bidirectional
        self.aggregation_type = aggregation_type
         
        # Create a (bidirectional) LSTM to encode sequence
        #self.encoder = nn.Linear(feat_size, emb_size, bias=False)
        if depth > 1:
             self.lstm = nn.LSTM(feat_size, hidden_size, batch_first=True, num_layers=depth,
                                    dropout=dropout,bidirectional=bidirectional)
        else:
            self.lstm = nn.LSTM(feat_size, hidden_size, batch_first=True, 
                            bidirectional=bidirectional)
        
 
        # The output of the LSTM doubles if we use a bidirectional encoder.
        # we have the forward and backward hidden states 
        encoding_size = hidden_size * 2 if bidirectional else hidden_size
        self.combination_layer = nn.Linear(encoding_size, encoding_size)
         
        # dropout layer for regularizetion of a sequence
        self.dropout_layer = nn.Dropout(p=dropout)
        #self.m = nn.LayerNorm(encoding_size)
        #self.softplus = nn.Softplus()
        self.relu = nn.ReLU()

    def forward(self, x, seq_mask=None, seq_len=None):
        # Encode the sentence using the LSTM.
        # [B, M, hid_size]
        #e = self.encoder(x)
        #e = self.dropout_layer(e)
        outputs, (final, _) = self.lstm(x)
        
        if self.aggregation_type == 'last_state':
            #print(h.size())
            if self.bidirectional:
                h_T_fwd = final[0]
                h_T_bwd = final[1]
                h = torch.cat([h_T_fwd, h_T_bwd], dim=-1)
            else:
                h = final[-1]
            h = self.relu(self.combination_layer(h))
            #h = self.dropout_layer(h)
        elif self.aggregation_type == 'mean':
            outputs = self.dropout_layer(outputs)
            #outputs = outputs.mean(dim=1)
            h = self.relu(self.combination_layer(outputs))
            #h = self.dropout_layer(h)
            #h = outputs
            h = h.mean(dim=1)
        elif self.aggregation_type == 'sum':
            outputs = self.dropout_layer(outputs)
            #outputs = otuputs.mean(dim=1)
            h = self.relu(self.combination_layer(outputs))
            #h = self.dropout_layer(h)
            #h = outputs
            h = h.sum(dim=1)
            
        #h = self.m(h)
        #h = self.dropout_layer(h)
        return h

class LSTMFeatExtractorPyro(nn.Module):
    def __init__(self, hidden_size,feat_size, emb_size,
                 bidirectional=False, dropout=0.2, aggregation_type='mean', depth=1, tag_size=1):
        """
        :param hidden_size: size of recurrent cell
        :param pad_idx: id of the -PAD- token
        :param bidirectional: use rnn of bidirectional rnn         
        """
        super().__init__()
        self.bidirectional = bidirectional
        self.aggregation_type = aggregation_type
         
        # Create a (bidirectional) LSTM to encode sequence
        #self.encoder = nn.Linear(feat_size, emb_size, bias=False)
        if depth > 1:
             self.lstm = nn.LSTM(feat_size, hidden_size, batch_first=True, num_layers=depth,
                                    dropout=dropout,bidirectional=bidirectional)
        else:
            self.lstm = nn.LSTM(feat_size, hidden_size, batch_first=True, 
                            bidirectional=bidirectional)
        
 
        # The output of the LSTM doubles if we use a bidirectional encoder.
        # we have the forward and backward hidden states 
        encoding_size = hidden_size * 2 if bidirectional else hidden_size
        self.combination_layer = nn.Linear(encoding_size, encoding_size)
         
        # dropout layer for regularizetion of a sequence
        self.dropout_layer = nn.Dropout(p=dropout)
        self.projection = nn.Linear(encoding_size, feat_size)
        #self.m = nn.LayerNorm(encoding_size)
        #self.softplus = nn.Softplus()
        self.relu = nn.ReLU()

    def forward(self, x, seq_mask=None, seq_len=None):
        # Encode the sentence using the LSTM.
        # [B, M, hid_size]
        #e = self.encoder(x)
        #e = self.dropout_layer(e)
        outputs, (final, _) = self.lstm(x)
        
        if self.aggregation_type == 'last_state':
            #print(h.size())
            if self.bidirectional:
                h_T_fwd = final[0]
                h_T_bwd = final[1]
                h = torch.cat([h_T_fwd, h_T_bwd], dim=-1)
            else:
                h = final[-1]
            h = self.relu(self.combination_layer(h))
            h = self.dropout_layer(h)
        elif self.aggregation_type == 'mean':
            outputs = self.dropout_layer(outputs)
            #outputs = outputs.mean(dim=1)
            h = self.relu(self.combination_layer(outputs))
            h = self.dropout_layer(h)
            #h = outputs
            h = h.mean(dim=1)
        elif self.aggregation_type == 'sum':
            outputs = self.dropout_layer(outputs)
            #outputs = otuputs.mean(dim=1)
            h = self.relu(self.combination_layer(outputs))
            h = self.dropout_layer(h)
            #h = outputs
            h = h.sum(dim=1)
            
        #h = self.m(h)
        h = self.projection(h)
        #h = self.dropout_layer(h)

        return h

