from __future__ import absolute_import
from __future__ import print_function
import codecs
from mimic3models import common_utils
import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader
from nltk.corpus import stopwords
from stop_words import get_stop_words
import string
from torch.optim.lr_scheduler import LambdaLR

class MIMICDataset(Dataset):
    """
       Loads time serie data into memory from a text file,
       split by newlines. 
    """
    def __init__(self, reader, discretizer, normalizer, target_repl=False, batch_labels=False):
        self.data = []
        self.y  = []
        N = reader.get_number_of_examples()
        #if small_part:
        #    N = 1000
        ret = common_utils.read_chunk(reader, N)
        data = ret["X"]
        ts = ret["t"]
        labels = ret["y"]
        names = ret["name"]
        data = [discretizer.transform(X, end=t)[0] for (X, t) in zip(data, ts)]
        if normalizer is not None:
            data = [normalizer.transform(X) for X in data]
        self.data = np.array(data, dtype=np.float32)
        self.T = self.data.shape[1]
        if batch_labels:
            self.y = np.array([[l] for l in labels], dtype=np.float32)
        else:
            self.y = np.array(labels, dtype=np.float32)
        if target_repl:
            self.y = self._extend_labels(self.y)

    def _extend_labels(self, labels):
        # (B,)
        labels = labels.repeat(self.T, axis=1)  # (B, T)
        return labels

    def __len__(self):
        # overide len to get number of instances
        return len(self.data)

    def __getitem__(self, idx):
        # get words and label for a given instance index
        return self.data[idx], self.y[idx]

def load_data(reader, discretizer, normalizer, small_part=False, return_names=False):
    N = reader.get_number_of_examples()
    if small_part:
        N = 1000
    ret = common_utils.read_chunk(reader, N)
    data = ret["X"]
    ts = ret["t"]
    labels = ret["y"]
    names = ret["name"]
    data = [discretizer.transform(X, end=t)[0] for (X, t) in zip(data, ts)]
    if normalizer is not None:
        data = [normalizer.transform(X) for X in data]
    whole_data = (np.array(data), labels)
    if not return_names:
        return whole_data
    return {"data": whole_data, "names": names}

UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"


class Vocabulary:
    """
        Creates a vocabulary from a word2vec file. 
    """
    def __init__(self):
        self.idx_to_word = {0: PAD_TOKEN, 1: UNK_TOKEN, 2: SOS_TOKEN, 3: EOS_TOKEN}
        self.word_to_idx = {PAD_TOKEN: 0, UNK_TOKEN: 1, SOS_TOKEN: 2, EOS_TOKEN: 3}
        self.word_freqs = {}
       
    
    def __getitem__(self, key):
        return self.word_to_idx[key] if key in self.word_to_idx else self.word_to_idx[UNK_TOKEN]
    
    def word(self, idx):
        return self.idx_to_word[idx]
    
    def size(self):
        return len(self.word_to_idx)
    
    
    def from_data(input_file, vocab_size, emb_size):
      
        vocab = Vocabulary()
        vocab_size = vocab_size + len(vocab.idx_to_word)
        weight = np.zeros((vocab_size, emb_size))
        with codecs.open(input_file, 'rb')  as f:
         
          for l in f:
            line = l.decode().split()
            token = line[0]
            if token not in vocab.word_to_idx:
              idx = len(vocab.word_to_idx)
              vocab.word_to_idx[token] = idx
              vocab.idx_to_word[idx] = token
            
              vect = np.array(line[1:]).astype(np.float)
              weight[idx] = vect
          # average embedding for unk word
          avg_embedding = np.mean(weight, axis=0)
          weight[1] = avg_embedding
                            
        return vocab, weight

class MIMICTextDataset(Dataset):
    """
       Loads a list of sentences into memory from a text file,
       split by newlines. 
    """
    def __init__(self, reader, discretizer, normalizer, 
            notes_output='sentence', max_w=25, max_s=500, max_d=500,
            target_repl=False, batch_labels=False):
        self.data = []
        self.y  = []
        self.max_w = max_w
        self.max_s = max_s
        self.max_d = max_d
        N = reader.get_number_of_examples()
        #if small_part:
        #    N = 1000
        ret = common_utils.read_chunk(reader, N)
        data = ret["X"]
        notes_text = ret["text"]
        notes_info = ret["text_info"]
        ts = ret["t"]
        labels = ret["y"]
        names = ret["name"]
        data = [discretizer.transform(X, end=t)[0] for (X, t) in zip(data, ts)]
        if normalizer is not None:
            data = [normalizer.transform(X) for X in data]
        #self.x = np.array(data, dtype=np.float32)
        #self.T = self.data.shape[1]
        #if target_repl:
        #    self.y = self._extend_labels(self.y)
        #notes into list of sentences, docs, etc..
        self.notes = []
        tmp_data = []
        tmp_labels = []
        if notes_output == 'sentence':
            # [N, W] patients, words
            for patient_notes, _x, l  in zip(notes_text, data, labels):
                tmp_notes = []
                for doc in sorted(patient_notes):
                    sentences = patient_notes[doc]
                    for sentence in sentences:
                        #print(sentence)
                        tmp_notes.extend(sentence)
                if len(tmp_notes) > 0 and len(tmp_notes) <= self.max_w:
                    #print(tmp_notes)
                    self.notes.append(' '.join(tmp_notes))
                    #self.notes.append(tmp_notes)
                    tmp_data.append(_x)
                    tmp_labels.append(l)
                #elif len(tmp_notes) > 0:
                #    self.notes.append(' '.join(tmp_notes[:self.max_w]))
                #    tmp_data.append(_x)
        elif notes_output == 'sentence-max':
             # [N, W] patients, words
             # [N, W] patients, words
            for patient_notes, _x, l  in zip(notes_text, data, labels):
                tmp_notes = []
                for doc in sorted(patient_notes):
                    sentences = patient_notes[doc]
                    for sentence in sentences:
                        #print(sentence)
                        tmp_notes.extend(sentence)
                if len(tmp_notes) > 0 and len(tmp_notes) <= self.max_w:
                    #print(tmp_notes)
                    self.notes.append(' '.join(tmp_notes))
                    #self.notes.append(tmp_notes)
                    tmp_data.append(_x)
                    tmp_labels.append(l)
                elif len(tmp_notes) > 0:
                    self.notes.append(' '.join(tmp_notes[:self.max_w]))
                    tmp_data.append(_x)
                    tmp_labels.append(l)

        elif notes_output == 'doc':
            # [N, S, W] patients, sentences, words
            # TODO add max size!
            for patient_notes,  _x, l in zip(notes_text, data, labels):
                tmp_notes = []
                for doc in sorted(patient_notes):
                    sentences = patient_notes[doc]
                    for sentence in sentences:
                        if len(sentence) > 0 and len(sentence) <= max_w:
                            tmp_notes.append(sentence)
                        elif len(sentence) > 0:
                            tmp_notes.append(sentence[:max_w])
                if len(tmp_notes) > 0 and len(tmp_notes) <= max_s:
                    self.notes.append(tmp_notes)
                    tmp_data.append(_x)
                    tmp_labels.append(l)
                elif len(tmp_notes) > 0:
                    self.notes.append(tmp_notes[:max_s])
                    tmp_data.append(_x)
                    tmp_labels.append(l)
        elif notes_output == 'doc-stopword':
            # [N, S, W] patients, sentences, words
            # TODO add max size!
            for patient_notes,  _x, l in zip(notes_text, data, labels):
                tmp_notes = []
                for doc in sorted(patient_notes):
                    sentences = patient_notes[doc]
                    for sentence in sentences:
                        sentence = _remove_stopwords(' '.join(sentence))
                        if len(sentence) > 0 and len(sentence) <= max_w:
                            tmp_notes.append(sentence)
                        elif len(sentence) > 0:
                            tmp_notes.append(sentence[:max_w])
                if len(tmp_notes) > 0 and len(tmp_notes) <= max_s:
                    self.notes.append(tmp_notes)
                    tmp_data.append(_x)
                    tmp_labels.append(l)
                elif len(tmp_notes) > 0:
                    self.notes.append(tmp_notes[:max_s])
                    tmp_data.append(_x)
                    tmp_labels.append(l)
        #elif notes_output == 'docs':
            # [N, D, S, W] patients, docs, sentences, words
            #for patient_notes, patient_notes_info, _x, l in zip(notes_text, notes_info, data, labels):
            #    tmp_notes = []
            #    doc_labels = []
            #    for doc in sorted(patient_notes):
            #        sentences = patient_notes[doc]
            #        doc_label = ' <hour=%s> '%(str(patient_notes_info[doc][3]))
            #        tmp_doc = []
            #        for sentence in sentences:
            #            if len(sentence) > 0 and len(sentence) <= max_w:
            #                tmp_doc.append(' '.join(sentence))
                    #tmp_notes.append(doc_label)
                    #if len(tmp_doc) > 0 and len(tmp_doc) <= max_s:
            #        tmp_notes.append(doc_label + ' <sent> '.join(tmp_doc))
                    
            #    if len(tmp_notes) > 0 and len(tmp_notes) <= max_d:
            #        self.notes.append(' <doc> '.join(tmp_notes))
            #        tmp_data.append(_x)
            #        tmp_labels.append(l)
#
        self.x = np.array(tmp_data, dtype=np.float32)   
        self.T = self.x.shape[1]
        if batch_labels:
            self.y = np.array([[l] for l in tmp_labels], dtype=np.float32)
        else:
            self.y = np.array(tmp_labels, dtype=np.float32)


    def _extend_labels(self, labels):
        # (B,)
        labels = labels.repeat(self.T, axis=1)  # (B, T)
        return labels

    def __len__(self):
        # overide len to get number of instances
        return len(self.x)

    def __getitem__(self, idx):
        # get words and label for a given instance index
        return self.x[idx], self.notes[idx], self.y[idx]

def _remove_stopwords(sent, lang='english'):
    # filter stopwords and punctuation
    sent = sent.translate(str.maketrans('', '', string.punctuation))
    word_list = sent.split()
    filtered_words = [word for word in word_list if word not in get_stop_words(lang)]
    return filtered_words


def doc_collate(batch):
    data = np.array([item[0] for item in batch])
    data = torch.tensor(data)
    notes = [item[1] for item in batch]
    target = np.array([item[2] for item in batch])
    target = torch.tensor(target)
    #target = torch.LongTensor(target)
    return [data, notes, target]


def create_sentence_batch(sentences, vocab, device, stopwords=False):
    """
    Converts a list of sentences to a padded batch of word ids. Returns
    an input batch, output tags, a sequence mask over
    the input batch, and a tensor containing the sequence length of each
    batch element.
    :param sentences: a list of sentences, each a list of token ids
    :param vocab: a Vocabulary object for this dataset
    :param device: 
    :returns: a batch of padded inputs,  mask, lengths
    """
    if stopwords:
        tok = [_remove_stopwords(sen) for sen in sentences]
    else:
        tok = [sen.split() for sen in sentences]
    #tok = np.array([sen[0] for sen in sentences])
    seq_lengths = [len(sen) for sen in tok]
    max_len = max(seq_lengths)
    pad_id = vocab[PAD_TOKEN]
    unk_id = vocab[UNK_TOKEN]

    pad_id_input = []
    #pad and find ids for words given the word2vec vocab
    #print(tok)
    for idx, sen in enumerate(tok):
      tmp_sent = []
      for t in range(max_len):
        if t < seq_lengths[idx]:
          try:
            token_id = vocab[sen[t]]
          except KeyError:
            token_id = unk_id
        else:
          token_id = pad_id
        tmp_sent.append(token_id)
      pad_id_input.append(tmp_sent) 

    
    # Convert everything to PyTorch tensors.
    batch_input = torch.tensor(pad_id_input)
    seq_mask = (batch_input != vocab[PAD_TOKEN])
    seq_length = torch.tensor(seq_lengths)
    
    # Move all tensors to the given device.
    batch_input = batch_input.to(device)
    seq_mask = seq_mask.to(device)
    seq_length = seq_length.to(device)
    
    return batch_input, seq_mask, seq_length

class SortingTextDataLoader:
    """
    A wrapper for the DataLoader class that sorts sentences by their
    lengths in descending order.
    """

    def __init__(self, dataloader):
        # we sort sentences for the optimization of the RNN code!
        self.dataloader = dataloader
        self.it = iter(dataloader)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        sentences = None
        labels = None
        data = None
        for x,s,l in self.it:
            data = x
            sentences = s
            labels = l
            break

        if data is None:
            self.it = iter(self.dataloader)
            raise StopIteration
        if sentences is None:
            self.it = iter(self.dataloader)
            raise StopIteration
        if labels is None:
            self.it = iter(self.dataloader)
            raise StopIteration
        data = np.array(data)
        sentences = np.array(sentences)
        labels = np.array(labels)
        sort_keys = sorted(range(len(sentences)), 
                           key=lambda idx: len(sentences[idx].split()), 
                           reverse=True)
        sorted_data = data[sort_keys]
        sorted_data = torch.tensor(sorted_data)
        sorted_sentences = sentences[sort_keys]
        #sorted_sentences = sorted_sentences.tolist()
        sorted_labels = labels[sort_keys]
        sorted_labels = torch.tensor(sorted_labels)
        return sorted_data, sorted_sentences, sorted_labels


def create_doc_batch(docs, vocab, device):
    """
    """

    sent_seq_lengths = np.array([len(doc) for doc in docs])
    word_seq_lengths = [[len(sent) for sent in doc] for doc in docs]
    b = len(docs)
    sent_max_len = max(sent_seq_lengths)
    word_max_len = max([max(w_seq) for w_seq in word_seq_lengths])
    pad_id = vocab[PAD_TOKEN]
    unk_id = vocab[UNK_TOKEN]

    pad_id_input = np.zeros((b, sent_max_len, word_max_len), dtype=int)
    word_seq_length = np.ones((b, sent_max_len), dtype=int)
    
    #for doc in docs:
    #    for sent in doc:
    #        word_seq_lengths.append(len(sent))
    #word_seq_lengths = np.asarray(word_seq_lengths)
    #word_max_len = max(word_seq_lengths)
    
    for i, w_lens in enumerate(word_seq_lengths):
        for j, w_len in enumerate(w_lens):
            word_seq_length[i][j] = w_len
    #pad and find ids for words given the word2vec vocab
    #print(tok)
    for idx_doc, doc in enumerate(docs):
        #tmp_doc = []
        for i in range(sent_max_len):
            tmp_sent = []
            if i < sent_seq_lengths[idx_doc]:
                sent = doc[i]
                for j in range(word_max_len):
                    if j < word_seq_lengths[idx_doc][i]: #[idx_doc][i]
                        try:
                            token_id = vocab[sent[j]]
                        except KeyError:
                            token_id = unk_id
                    else:
                        token_id = pad_id
                    #tmp_sent.append(token_id)
                    pad_id_input[idx_doc][i][j] = token_id
            else:
                #tmp_sent = [pad_id for _ in range(word_max_len[idx_doc])] 
                for j in range(word_max_len):
                    pad_id_input[idx_doc][i][j] = pad_id
        #pad_id_input.append(tmp_sent) 

    # Convert everything to PyTorch tensors.
    batch_input = torch.tensor(pad_id_input)
    #seq_mask = (batch_input != vocab[PAD_TOKEN])
    sent_seq_length = torch.tensor(sent_seq_lengths)
    word_seq_length = torch.tensor(word_seq_length.flatten())
    
    # Move all tensors to the given device.
    batch_input = batch_input.to(device)
    #seq_mask = seq_mask.to(device)
    sent_seq_length = sent_seq_length.to(device)
    word_seq_length = word_seq_length.to(device)
    
    return batch_input, sent_seq_length, word_seq_length


def create_doc_batch_mask(docs, vocab, device):
    """
    """
    sent_seq_lengths = np.array([len(doc) for doc in docs])
    word_seq_lengths = [[len(sent) for sent in doc] for doc in docs]
    b = len(docs)
    sent_max_len = max(sent_seq_lengths)
    word_max_len = max([max(w_seq) for w_seq in word_seq_lengths])
    pad_id = vocab[PAD_TOKEN]
    unk_id = vocab[UNK_TOKEN]

    pad_id_input = np.zeros((b, sent_max_len, word_max_len), dtype=int)
    word_seq_length = np.ones((b, sent_max_len), dtype=int)

    # for doc in docs:
    #    for sent in doc:
    #        word_seq_lengths.append(len(sent))
    # word_seq_lengths = np.asarray(word_seq_lengths)
    # word_max_len = max(word_seq_lengths)

    for i, w_lens in enumerate(word_seq_lengths):
        for j, w_len in enumerate(w_lens):
            word_seq_length[i][j] = w_len
    # pad and find ids for words given the word2vec vocab
    # print(tok)
    for idx_doc, doc in enumerate(docs):
        # tmp_doc = []
        for i in range(sent_max_len):
            tmp_sent = []
            if i < sent_seq_lengths[idx_doc]:
                sent = doc[i]
                for j in range(word_max_len):
                    if j < word_seq_lengths[idx_doc][i]:  # [idx_doc][i]
                        try:
                            token_id = vocab[sent[j]]
                        except KeyError:
                            token_id = unk_id
                    else:
                        token_id = pad_id
                    # tmp_sent.append(token_id)
                    pad_id_input[idx_doc][i][j] = token_id
            else:
                # tmp_sent = [pad_id for _ in range(word_max_len[idx_doc])]
                for j in range(word_max_len):
                    pad_id_input[idx_doc][i][j] = pad_id
        # pad_id_input.append(tmp_sent)

    # Convert everything to PyTorch tensors.

    #pad_id_input_slice = pad_id_input[:, :, 0]
    #batch_input_slice = torch.tensor(pad_id_input_slice)

    batch_input = torch.tensor(pad_id_input)
    seq_mask = (batch_input[:, :, 0] != vocab[PAD_TOKEN])
    sent_seq_length = torch.tensor(sent_seq_lengths)
    word_seq_length = torch.tensor(word_seq_length.flatten())

    # Move all tensors to the given device.
    batch_input = batch_input.to(device)
    seq_mask = seq_mask.to(device)
    sent_seq_length = sent_seq_length.to(device)
    word_seq_length = word_seq_length.to(device)

    return batch_input, sent_seq_length, seq_mask, word_seq_length

def create_doc_batch_bert(docs, tokenizer, device):
    """
    """
    sent_seq_lengths = np.array([len(doc) for doc in docs])
    word_seq_lengths = [[len(sent) for sent in doc] for doc in docs]
    b = len(docs)
    sent_max_len = max(sent_seq_lengths)
    word_max_len = max([max(w_seq) for w_seq in word_seq_lengths])
    pad_id = tokenizer.pad_token_id
    

    pad_id_input = np.zeros((b, sent_max_len, word_max_len), dtype=int)
    word_seq_length = np.ones((b, sent_max_len), dtype=int)
    
    #for doc in docs:
    #    for sent in doc:
    #        word_seq_lengths.append(len(sent))
    #word_seq_lengths = np.asarray(word_seq_lengths)
    #word_max_len = max(word_seq_lengths)
    
    for i, w_lens in enumerate(word_seq_lengths):
        for j, w_len in enumerate(w_lens):
            word_seq_length[i][j] = w_len
    #pad and find ids for words given the word2vec vocab
    #print(tok)
    for idx_doc, doc in enumerate(docs):
        #tmp_doc = []
        for i in range(sent_max_len):
            tmp_sent = []
            #TODO ensure that is not larger than bert size
            if i < sent_seq_lengths[idx_doc]:
                #input to tokenizer str
                sent = ' '.join(doc[i])
                #BERT tokenizer over a sentence [CLS] w1, w2, ... [SEP]
                #truncates up to max bert
                sent_tok = tokenizer(sent, truncation=True, add_special_tokens=True) 
                sent_id = sent_tok['input_ids']
                for j in range(word_max_len):
                    if j < word_seq_lengths[idx_doc][i]: #[idx_doc][i]
                        token_id = sent_id[j]
                    else:
                        token_id = pad_id
                    #tmp_sent.append(token_id)
                    pad_id_input[idx_doc][i][j] = token_id
            else:
                #tmp_sent = [pad_id for _ in range(word_max_len[idx_doc])] 
                for j in range(word_max_len):
                    pad_id_input[idx_doc][i][j] = pad_id
        #pad_id_input.append(tmp_sent) 

    # Convert everything to PyTorch tensors.
    batch_input = torch.tensor(pad_id_input)
    #seq_mask = (batch_input != vocab[PAD_TOKEN])
    sent_seq_length = torch.tensor(sent_seq_lengths)
    word_seq_length = torch.tensor(word_seq_length.flatten())
    
    # Move all tensors to the given device.
    batch_input = batch_input.to(device)
    #seq_mask = seq_mask.to(device)
    sent_seq_length = sent_seq_length.to(device)
    word_seq_length = word_seq_length.to(device)
    
    return batch_input, sent_seq_length, word_seq_length


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def create_seq_batch_bert(sentences, max_bert_len, tokenizer, device):
    """
    """
    tok = [sen.split() for sen in sentences]
    # tok = np.array([sen[0] for sen in sentences])
    seq_lengths = [len(sen) for sen in tok]
    max_len = max(seq_lengths)

    pad_id = tokenizer.pad_token_id

    #pad_id_input = np.zeros((b, sent_max_len, word_max_len), dtype=int)
    #word_seq_length = np.ones((b, sent_max_len), dtype=int)

    pad_id_input = []
    # pad and find ids for words given the word2vec vocab
    # print(tok)
    for idx, sen in enumerate(tok):
        sent_tok = tokenizer(sen, add_special_tokens=True)
        sent_id = sent_tok['input_ids']
        #for t in range(max_len):

        pad_id_input.append()

        # Convert everything to PyTorch tensors.
    batch_input = torch.tensor(pad_id_input)

    seq_length = torch.tensor(seq_lengths)

    # Move all tensors to the given device.
    batch_input = batch_input.to(device)

    seq_length = seq_length.to(device)

    # Move all tensors to the given device.
    batch_input = batch_input.to(device)
    # seq_mask = seq_mask.to(device)
    #sent_seq_length = sent_seq_length.to(device)
    #word_seq_length = word_seq_length.to(device)

    return batch_input



def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0,
    after a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def save_results(names, pred, y_true, path):
    common_utils.create_directory(os.path.dirname(path))
    with open(path, 'w') as f:
        f.write("stay,prediction,y_true\n")
        for (name, x, y) in zip(names, pred, y_true):
            f.write("{},{:.6f},{}\n".format(name, x, y))
