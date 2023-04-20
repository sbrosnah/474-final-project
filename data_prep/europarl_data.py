import os 
import subprocess
import pandas as pd 
import csv
from data_aligner import DataAligner
from bert_embedder import BertEmbedder

# The list of currently supported languages and their codes are as follows:
#
# bg - Bulgarian
# cs - Czech
# da - Danish
# de - German
# el - Greek
# es - Spanish
# et - Estonian
# fi - Finnish
# fr - French
# hu - Hungarian
# it - Italian
# lt - Lithuanian
# lv - Latvian
# nl - Dutch
# pl - Polish
# pt - Portuguese
# ro - Romanian
# sk - Slovak
# sl - Slovene
# sv - Swedish

class EuroParlData:
    def __init__(self, redo):
        self.languages = ["bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "hu", "it", "lt", "lv", "nl", "pl", "pt", "ro", "sk", "sl", "sv"]
        self.root_dir = os.path.join(os.getcwd(), "europarl-data")
        self.txt_dir = os.path.join(self.root_dir, "txt")
        self.tokenized_dir = os.path.join(self.root_dir, "tokenized-files")
        self.combined_dir = os.path.join(self.root_dir, "combined-files")
        self.split_dir = os.path.join(self.root_dir, "split-files")
        self.cleaned_dir = os.path.join(self.root_dir, "cleaned-files")
        self.csv_dir = os.path.join(self.root_dir, "csv-files")
        self.split_sentence_script = os.path.join(os.getcwd(), "europarl-data", "tools", "split-sentences.perl")
        self.tokenize_script = os.path.join(os.getcwd(), "europarl-data", "tools", "tokenizer.perl")
        self.redo_all = redo

        batch_size = 300
        save_multiple = 1000
        clean_multiple = 30
        start_where_left = True

        self.embedder = BertEmbedder(batch_size, save_multiple, clean_multiple, start_where_left)
        self.aligner = DataAligner()
    
    def get_tokenized_file(self, language_code):
        return os.path.join(self.tokenized_dir, f"{language_code}-tokenized.txt")
    
    def get_combined_file(self, language_code):
        return os.path.join(self.combined_dir, f"{language_code}-combined.txt")
    
    def get_split_file(self, language_code):
        return os.path.join(self.split_dir, f"{language_code}-split.txt")
    
    def get_cleaned_file(self, language_code):
        return os.path.join(self.cleaned_dir, f"{language_code}-cleaned.txt")

    def get_csv_file(self, language_code):
        return os.path.join(self.csv_dir, f"{language_code}.csv")
    
    def get_embedding_file(self, language_code):
        return os.path.join(self.csv_dir, f"{language_code}.pt")
    
    def get_filepaths(self, language_code):
        f = []
        for (dirpath, dirnames, filenames) in os.walk(os.path.join(self.txt_dir, language_code)):
            f.extend([os.path.join(dirpath, f) for f in filenames])
        return f 
    


    def combine_files(self, language_code):
        in_fps = self.get_filepaths(language_code)
        output_file = self.get_combined_file(language_code)

        #check if it exists 
        if not self.redo_all and os.path.exists(output_file):
            return

        with open(output_file, "w+") as of:
            for in_fp in in_fps:
                with open(in_fp, "r") as in_f:
                    of.write(in_f.read())

    def combine_files(self, language_code):
        in_fps = self.get_filepaths(language_code)
        output_file = self.get_combined_file(language_code)

        if not self.redo_all and os.path.exists(output_file):
            return
        
        with open(output_file, "w+") as of:
            for in_fp in in_fps:
                with open(in_fp, "r") as in_f:
                    of.write(in_f.read())
    
    def run_sript(self, script_path, language_code, input_file, output_file):
        cmd = [script_path, '-l', language_code]
        with open(output_file, 'w+') as f:
            subprocess.run(cmd, stdin=open(input_file), stdout=f)

    def split_file(self, language_code):
        input_file = self.get_combined_file(language_code)
        output_file = self.get_split_file(language_code)

        if not self.redo_all and os.path.exists(output_file):
            return
        
        self.run_sript(self.split_sentence_script, language_code, input_file, output_file)


    def tokenize_file(self, language_code):
        input_file = self.get_split_file(language_code)
        output_file = self.get_tokenized_file(language_code)

        if not self.redo_all and os.path.exists(output_file):
            return
        
        self.run_sript(self.tokenize_script, language_code, input_file, output_file)

    def make_line_deletions(self, line):

        if line[0] == "<":
            return "" 
        
        #convert line to lowercase 
        line = line.strip()
        line = line.lower()
        
        tokens = line.split(" ")
        done = False 
        while not done:
            done = True 
            if len(tokens) > 0 and tokens[0].replace(".", "").isnumeric():
                del tokens[0]
                done = False
            if len(tokens) > 0 and tokens[0] == ".":
                del tokens[0]
                done = False
            if len(tokens) > 0 and (tokens[0] == '-' or tokens[0] == '–'):
                del tokens[0]
                done = False
            if len(tokens) > 0 and tokens[0] == "(":
                #for the case {digit} . ( 
                if len(tokens) == 1:
                    del tokens[0]
                    done = False
                #for the case ( {language code} )
                else: 
                    #find index of next paren 
                    ind = 1
                    for t in range(1, len(tokens)):
                        #if one wasn't found, don't try and delete anything yet 
                        if t == len(tokens) - 1 and tokens[t] != ")":
                            ind = -1
                            break

                        if tokens[t] == ")":
                            ind = t 
                            break 
                    
                    if ind > 0:
                        #delete all of those tokens 
                        tokens = tokens[ind + 1:]
                        done = False
                    else:
                        tokens = tokens[1:]
                        done = False
                        
            if len(tokens) > 0 and tokens[0] == ",":
                del tokens[0]
                done = False
            
            if len(tokens) > 0 and tokens[0] == ")":
                del tokens[0]

        if len(tokens) > 0 and tokens[-1] == "(":
            del tokens[-1]
        
        
        return " ".join(tokens)

    def clean_lines(self, language_code):
        #loop through lines 
        #lower case it 
        #make the deletions 
        read_file_path = self.get_tokenized_file(language_code)
        write_file_path = self.get_cleaned_file(language_code)

        if not self.redo_all and os.path.exists(write_file_path):
            return

        with open(read_file_path, "r") as rf:
            with open(write_file_path, "w+") as wf:
                for line in rf:
                    line = self.make_line_deletions(line)
                    if line != "":
                        wf.write(line + "\n")
    
    def create_csv(self, language_code):
        input_file = self.get_cleaned_file(language_code)
        output_file = self.get_csv_file(language_code)

        if not self.redo_all and os.path.exists(output_file):
            return 
        
        with open(input_file, "r") as f:
            rows = []
            for line in f:
                line = line.strip()
                rows.append(line)
        df = pd.DataFrame(rows, index=None)
        df.to_csv(output_file)
    
    def create_embeddings(self, language_code):

        save_path = self.get_embedding_file(language_code)

        if not self.redo_all and os.path.exists(save_path):
            return
        
        sentence_list = self.get_language_list(language_code)
        embeddings = self.embedder.get_sentence_embeddings(sentence_list)
        self.embedder.save_embeddings(embeddings, save_path)
    
    def get_language_df(self, language_code):
        file_path = self.get_csv_file(language_code)

        if not os.path.exists(file_path):
            return None
        
        df = pd.read_csv(file_path)

        return df
    
    def get_language_list(self, language_code):

        file_path = self.get_cleaned_file(language_code)

        if not os.path.exists(file_path):
            return None
        
        with open(file_path, "r", encoding="utf-8") as f:
            rows = []
            for line in f:
                line = line.strip()
                rows.append(line)

        return rows

    def prep_language(self, language_code):
        self.combine_files(language_code)
        self.split_file(language_code)
        self.tokenize_file(language_code)
        self.clean_lines(language_code)
        self.create_csv(language_code)
        self.create_embeddings(language_code)

    def prep_languages(self):
        for language_code in self.languages:
            self.prep_language(language_code)

if __name__=="__main__":
    prepper = EuroParlData(False)
    prepper.prep_language("en")