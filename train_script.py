from mt_transformer_trainer import MtTransformerTrainer, DataLoader, Translate
from transformer import TransformerModel
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

data_loader = DataLoader()
trainer = MtTransformerTrainer(device)
translator = Translate(device)

src_lines = ["hi", "bye"]
tgt_lines = ["hola", "nos vemos"]

SRC, TGT, train, val = data_loader.prep_data(src_lines, tgt_lines)
pad_idx = TGT.vocab.stoi["<blank>"]
model = TransformerModel(len(SRC.vocab), len(TGT.vocab), N=2).to(device)
trainer.train_model(model, TGT, train, val)
translator.execute(train, val, SRC, TGT, model)