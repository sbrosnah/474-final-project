import torch
from transformers import BertTokenizer, BertModel

class BertEmbedder:
    def __init__(self):
        #determine the device 
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-uncased")
        self.model = BertModel.from_pretrained("bert-base-multilingual-uncased").to(self.device)
        # Put the model in "evaluation" mode, meaning feed-forward operation.
        self.model.eval()
        self.batch_size = 64

    def get_sentence_embeddings(self, sentence_list):

        embeddings = []

        #We process this in batches to speed things up as much as we can
        for i in range(0, len(sentence_list), self.batch_size):
            batch_sentences = sentence_list[i:i+self.batch_size]   
            batch_tokens = self.tokenizer.batch_encode_plus(batch_sentences, return_tensors="pt", truncation=True, padding=True)
            batch_tokens = batch_tokens.to(self.device)
            
            with torch.no_grad():
                outputs = self.model(batch_tokens['input_ids'], attention_mask=batch_tokens['attention_mask'], token_type_ids=batch_tokens['token_type_ids'])

            last_hidden_states = outputs.last_hidden_state
            last_hidden_states = last_hidden_states.to(self.device)

            # # Calculate the average of all 23 token vectors.
            sentence_embeddings = torch.mean(last_hidden_states, dim=1)

            embeddings.append(sentence_embeddings)

        embeddings = torch.cat(embeddings, dim=0)
        embeddings = embeddings.to(self.device)

        return embeddings
    
    def save_embeddings(self, embeddings, filepath):
        #embeddings should be a tensor and filepath should be to a file with a .pt extension 
        torch.save(embeddings, filepath)
    
    def load_embeddings(self, filepath):
        embeddings = torch.load(filepath)
        return embeddings

