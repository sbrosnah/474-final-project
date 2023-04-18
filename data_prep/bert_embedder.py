import torch
from transformers import AutoTokenizer, BertModel

class BertEmbedder:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        self.model = BertModel.from_pretrained("bert-base-multilingual-cased")
        # Put the model in "evaluation" mode, meaning feed-forward operation.
        self.model.eval()

    def get_sentence_embedding(self, text):

        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**inputs)

        last_hidden_states = outputs.last_hidden_state

        # # Remove dimension 1, the "batches".
        token_embeddings = torch.squeeze(last_hidden_states, dim=1)

        # # Swap dimensions 0 and 1.
        token_embeddings = token_embeddings.permute(1,0,2)

        # # Calculate the average of all 23 token vectors.
        sentence_embedding = torch.mean(token_embeddings, dim=0)

        return sentence_embedding.detach()[0]

    def get_sentence_embeddings(self, sentence_list):
        size = len(sentence_list)
        interval = size // 100
        counter = 0

        sentence_embeddings = []
        for sentence in sentence_list:
            if counter % interval == 0:
                percentage_done = counter // interval 
                print(f"{percentage_done}% done!")
            sentence_embeddings.append(self.get_sentence_embedding(sentence))
            counter += 1
        

        
        return torch.cat(sentence_embeddings)
    
    def save_embeddings(self, embeddings, filepath):
        #embeddings should be a tensor and filepath should be to a file with a .pt extension 
        torch.save(embeddings, filepath)
    
    def load_embeddings(self, filepath):
        embeddings = torch.load(filepath)
        return embeddings

