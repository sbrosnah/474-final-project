import torch
from transformers import BertTokenizer, BertModel
import gc
import os
import shutil

class BertEmbedder:
    def __init__(self, batch_size, save_multiple, clean_multiple, start_where_left):
        #determine the device 
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-uncased")
        self.model = BertModel.from_pretrained("bert-base-multilingual-uncased").to(self.device)
        # Put the model in "evaluation" mode, meaning feed-forward operation.
        self.model.eval()

        self.batch_size = batch_size
        self.save_size = save_multiple * batch_size
        self.clean_size = clean_multiple * batch_size
        self.root_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_embeddings")
        self.start_where_left = start_where_left
    
    def restore_state(self, sentence_list):
        #get all of the paths in the directory
        paths = [os.path.join(self.root_save_path, n) for n in os.listdir(self.root_save_path)]
        max_count = -1
        count = 0
        for p in paths:
            name = os.path.basename(p)
            parts = name.split('-')
            count = int(parts[2].split(".")[0])
            max_count = max(count, max_count)
        
        if max_count != -1:
            start = (max_count + 1) * self.save_size
        else:
            start = 0

        #update sentence list
        sentence_list = sentence_list[start:]

        #return the max_count + 1 and the paths 
        return max_count + 1, paths, sentence_list

    def get_sentence_embeddings(self, sentence_list):

        embeddings = []
        save_count = 0
        save_paths = []

        if self.start_where_left:
            #TODO: change save_count to be where we left off, verify sizes of loaded up vectors, and cut list down appropriately
            save_count, save_paths, sentence_list = self.restore_state(sentence_list)


        #We process this in batches to speed things up as much as we can
        for i in range(0, len(sentence_list), self.batch_size):
            batch_sentences = sentence_list[i:i+self.batch_size]   
            batch_tokens = self.tokenizer.batch_encode_plus(batch_sentences, return_tensors="pt", truncation=True, padding=True, )
            batch_tokens = batch_tokens.to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**batch_tokens)

                last_hidden_states = outputs.last_hidden_state
                del outputs
                last_hidden_states = last_hidden_states.cpu()

                # # Calculate the average of all 23 token vectors.
                # sentence_embeddings = torch.mean(last_hidden_states, dim=1).detach()
                attention_mask = batch_tokens['attention_mask'].cpu()
                masked_last_hidden_state = last_hidden_states * attention_mask.unsqueeze(-1)
                sentence_embeddings = masked_last_hidden_state.sum(1) / attention_mask.sum(1).unsqueeze(-1)
                del last_hidden_states

                embeddings.append(sentence_embeddings)
            
            del batch_tokens 

            if (((i+self.batch_size) % self.clean_size) == 0):
                gc.collect()
                if self.device is torch.device("cuda"):
                    torch.cuda.empty_cache()

            if (((i+self.batch_size) % self.save_size) == 0):
                embeddings = torch.cat(embeddings, dim=0)
                p = os.path.join(self.root_save_path, f"tmp-embeddings-{save_count}.pt")
                self.save_embeddings(embeddings, p)
                save_paths.append(p)
                save_count += 1
                embeddings = []
            
            print(f"sentences done: {i + 1}/{len(sentence_list)}")

        if len(embeddings) != 0:
            embeddings = torch.cat(embeddings, dim=0)
            p = os.path.join(self.root_save_path, f"tmp-embeddings-{save_count}.pt")
            self.save_embeddings(embeddings, p)
            save_paths.append(p)
        else: 
            save_count -= 1

        #now we loop through all of the save paths, load them up into embeddings, concatenate it, and then return it
        embeddings= []
        for save_path in save_paths:
            e = self.load_embeddings(save_path)
            embeddings.append(e)

        embeddings = torch.cat(embeddings, dim=0)

        #delete all of the temp files 
        self.delete_temp_files()

        return embeddings

    def delete_temp_files(self):
        for filename in os.listdir(self.root_save_path):
            file_path = os.path.join(self.root_save_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    
    def save_embeddings(self, embeddings, filepath):
        #embeddings should be a tensor and filepath should be to a file with a .pt extension 
        torch.save(embeddings, filepath)
    
    def load_embeddings(self, filepath):
        embeddings = torch.load(filepath)
        return embeddings

