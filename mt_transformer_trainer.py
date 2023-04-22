import numpy as np
import torch
import torch.nn as nn
import time
from torch.autograd import Variable
import seaborn
seaborn.set_context(context="talk")
from torchtext.legacy import data
import gc
import spacy

def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        out = model.decode(memory, src_mask, 
                        Variable(ys), 
                        Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, 
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys

class MtTransformerTrainer:
    def __init__(self, device):
        self.device = device


    def run_epoch(self, data_iter, model, loss_compute):
        "Standard Training and Logging Function"
        start = time.time()
        total_tokens = 0
        total_loss = 0
        tokens = 0
        for i, batch in enumerate(data_iter):
            out = model.forward(batch.src, batch.trg, 
                                batch.src_mask, batch.trg_mask)
            loss = loss_compute(out, batch.trg_y, batch.ntokens)
            total_loss += loss
            total_tokens += batch.ntokens
            tokens += batch.ntokens
            if i % 50 == 1:
                elapsed = time.time() - start
                print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                        (i, loss / batch.ntokens, tokens / elapsed))
                start = time.time()
                tokens = 0
        return total_loss / total_tokens

    def rebatch(self, pad_idx, batch):
        "Fix order in torchtext to match ours"
        src, trg = batch.src.transpose(0, 1).to(self.device), batch.trg.transpose(0, 1).to(self.device)
        return Batch(src, trg, pad_idx)

    def train_model(self, model, TGT, train, val):

        pad_idx = TGT.vocab.stoi["<blank>"]

        gc.collect()

        n_epochs = 10

        criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
        criterion.to(self.device)
        BATCH_SIZE = 1000
        train_iter = DataIterator(train, batch_size=BATCH_SIZE, device=self.device,
                                repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                                batch_size_fn=batch_size_fn, train=True)
        valid_iter = DataIterator(val, batch_size=BATCH_SIZE, device=self.device,
                                repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                                batch_size_fn=batch_size_fn, train=False)

        model_opt = torch.optim.Adam(model.parameters(), lr=5e-4)
        for epoch in range(n_epochs):
            model.train()
            self.run_epoch((self.rebatch(pad_idx, b) for b in train_iter), 
                    model, 
                    LossFunction(model.generator, criterion, model_opt))
            model.eval()

class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    
    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask

class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))
    
class LossFunction:
    "A simple loss compute and train function."
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), 
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.zero_grad()
        return loss.data * norm



class DataIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b
            self.batches = pool(self.data(), self.random_shuffler)
            
        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size,
                                          batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))

class DataLoader:
    def __init__(self):
        # Load spacy tokenizers.
        self.spacy_es = spacy.load('es_core_news_sm')
        self.spacy_en = spacy.load('en_core_web_sm')

    def tokenize_es(self, text):
        return [tok.text for tok in self.spacy_es.tokenizer(text)]

    def tokenize_en(self, text):
        return [tok.text for tok in self.spacy_en.tokenizer(text)]

    def prep_data(self, src_lines, tgt_lines):
        BOS_WORD = '<s>'
        EOS_WORD = '</s>'
        BLANK_WORD = "<blank>"
        SRC = data.Field(tokenize=self.tokenize_es, pad_token=BLANK_WORD)
        TGT = data.Field(tokenize=self.tokenize_en, init_token = BOS_WORD, 
                        eos_token = EOS_WORD, pad_token=BLANK_WORD)


        fields = (["src", SRC], ["trg", TGT])
        examples = [data.Example.fromlist((tgt_lines[i], src_lines[i]), fields ) for i in range(len(src_lines))]

        MAX_LEN = 200
        train, val = data.Dataset(examples, fields=fields, filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and 
                len(vars(x)['trg']) <= MAX_LEN).split()

        MIN_FREQ = 1
        SRC.build_vocab(train.src, min_freq=MIN_FREQ)
        TGT.build_vocab(train.trg, min_freq=MIN_FREQ)

        return SRC, TGT, train, val
    
class Translate:
    def __init__(self, device):
        self.device = device
    
    def execute(self, train, val, SRC, TGT, model):
        BATCH_SIZE = 1000
        n_train_iters = len(train) / BATCH_SIZE
        valid_iter = DataIterator(val, batch_size=BATCH_SIZE, device=self.device,
                                repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                                batch_size_fn=batch_size_fn, train=False)
            
        for i, batch in enumerate(valid_iter):
            src = batch.src.transpose(0, 1)[:1].to(self.device)
            src_mask = (src != SRC.vocab.stoi["<blank>"]).unsqueeze(-2).to(self.device)
            out = greedy_decode(model, src, src_mask, 
                                max_len=60, start_symbol=TGT.vocab.stoi["<s>"])
            print("Spanish:", end="\t")
            for i in range(0, src.size(1)):
                sym = SRC.vocab.itos[src[0, i]]
                if sym == "</s>": break
                print(sym, end =" ")
            print()
            print("Translation:", end="\t")
            for i in range(1, out.size(1)):
                sym = TGT.vocab.itos[out[0, i]]
                if sym == "</s>": break
                print(sym, end =" ")
            print()
            print("Target:\t", end="\t")
            for i in range(1, batch.trg.size(0)):
                sym = TGT.vocab.itos[batch.trg.data[i, 0]]
                if sym == "</s>": break
                print(sym, end =" ")
            print()
            print()
            
            if i > 1000 and i<1100:
                break


    
