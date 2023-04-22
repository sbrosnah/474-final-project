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
        self.combined_dir = os.path.join(self.root_dir, "combined-files")
        self.split_dir = os.path.join(self.root_dir, "split-files")
        self.cleaned_dir = os.path.join(self.root_dir, "cleaned-files")
        self.embedding_dir = os.path.join(self.root_dir, "embedding-files")
        self.aligned_dir = os.path.join(self.root_dir, "aligned-files")
        self.split_sentence_script = os.path.join(os.getcwd(), "europarl-data", "tools", "split-sentences.perl")
        self.redo_all = redo

        batch_size = 30
        save_multiple = 5
        clean_multiple = 2
        start_where_left = True
        align_batch_size = 5

        self.embedder = BertEmbedder(batch_size, save_multiple, clean_multiple, start_where_left)
        self.aligner = DataAligner(align_batch_size)
    
    def cleaned_filename(self, language_code, i):
        return f"{language_code}-cleaned-file-{i}.txt"

    def embedding_filename(self, language_code, i):
        return f"{language_code}-embedding-file-{i}.pt"
    
    def split_filename(self, language_code, i):
        return f"{language_code}-split-file-{i}.txt"
    
    def aligned_sentence_filename(self, language_code_one, language_code_two, i):
        return f"{language_code_one}-{language_code_two}-aligned-sentence-file-{i}.txt"

    def aligned_embedding_filename(self, language_code_one, language_code_two, i):
        return f"{language_code_one}-{language_code_two}-aligned-embedding-file-{i}.txt"
    
    def get_file(self, strategy):

        if not os.path.exists(strategy.get_file()):
            return None 

        return strategy.get_file()   
    
    def get_cleaned_file(self, language_code, i):

        class Strategy:
            def __init__(self, outer):
                self.outer = outer

            def get_file(self):
                return os.path.join(self.outer.cleaned_dir, language_code, self.outer.cleaned_filename(language_code, i))
            
        strategy = Strategy(self)
        return self.get_file(strategy)
    
    def get_embedding_file(self, language_code, i):

        class Strategy:
            def __init__(self, outer):
                self.outer = outer

            def get_file(self):
                return os.path.join(self.outer.embedding_dir, language_code, self.outer.embedding_filename(language_code, i))
            
        strategy = Strategy(self)
        return self.get_file(strategy)

    def get_aligned_file_pair(self, language_code_one, language_code_two, i):
        p = f"{language_code_one}-{language_code_two}"
        sent = os.path.join(self.aligned_dir, p, "sentences", self.aligned_sentence_filename(language_code_one, language_code_two, i))
        emb = os.path.join(self.aligned_dir, p, "embeddings", self.aligned_embedding_filename(language_code_one, language_code_two, i))
        
        return sent, emb
    
    def get_files(self, d):
        files = []

        if not os.path.exists(d):
            return files
        
        for filename in os.listdir(d):
            files.append(os.path.join(d, filename))

        files.sort()

        return files
    
    def get_split_files(self, language_code):
        d = os.path.join(self.split_dir, language_code)
        return self.get_files(d)
    
    def get_cleaned_files(self, language_code):
        d = os.path.join(self.cleaned_dir, language_code)
        return self.get_files(d)
    
    def get_embedding_files(self, language_code):
        d = os.path.join(self.embedding_dir, language_code)
        return self.get_files(d)
    
    def get_txt_files(self, language_code):
        d = os.path.join(self.txt_dir, language_code)
        return self.get_files(d)
    
    def get_aligned_file_pairs(self, language_code_one, language_code_two):
        p = f"{language_code_one}-{language_code_two}"
        d = os.path.join(self.aligned_dir, p, "sentences")
        sentence_files = self.get_files(d)
        d = os.path.join(self.aligned_dir, p, "embeddings")
        embedding_files = self.get_files(d)
        return list(zip(sentence_files, embedding_files))
    
    def run_sript(self, script_path, language_code, input_file, output_file):
        cmd = [script_path, '-l', language_code]
        with open(output_file, 'w+') as f:
            subprocess.run(cmd, stdin=open(input_file), stdout=f)
    
    def do_operation(self, strategy):

        input_files = strategy.get_input_files()
        output_files = []
        d = strategy.get_dir()

        if not os.path.exists(d):
            os.makedirs(d)

        for i in range(len(input_files)):
            output_files.append(os.path.join(d, strategy.get_file_name(i)))
        
        for in_file, out_file in zip(input_files, output_files):

            if not self.redo_all and os.path.exists(out_file):
                continue
            
            strategy.do_op(in_file, out_file)

    def split_files(self, language_code):

        class Strategy:
            def __init__(self, outer):
                self.outer = outer

            def get_input_files(self):
                return self.outer.get_txt_files(language_code)
            def get_dir(self):
                return os.path.join(self.outer.split_dir, language_code)
            def get_file_name(self, i):
                return self.outer.split_filename(language_code, i)
            def do_op(self, in_file, out_file):
                self.outer.run_sript(self.outer.split_sentence_script, language_code, in_file, out_file)

        strategy = Strategy(self)
        self.do_operation(strategy)

    def clean_lines(self, language_code):

        class Strategy:
            def __init__(self, outer):
                self.outer = outer

            def get_input_files(self):
                return self.outer.get_split_files(language_code)
            def get_dir(self):
                return os.path.join(self.outer.cleaned_dir, language_code)
            def get_file_name(self, i):
                return self.outer.cleaned_filename(language_code, i)
            def do_op(self, in_file, out_file):
                with open(in_file, "r") as rf:
                    with open(out_file, "w+") as wf:
                        for line in rf:
                            line = self.outer.make_line_deletions(line)
                            if line != "":
                                wf.write(line + "\n")
        strategy = Strategy(self)
        self.do_operation(strategy)
    
    def create_embeddings(self, language_code):

        class Strategy:
            def __init__(self, outer):
                self.outer = outer

            def get_input_files(self):
                return self.outer.get_cleaned_files(language_code)
            def get_dir(self):
                return os.path.join(self.outer.embedding_dir, language_code)
            def get_file_name(self, i):
                return self.outer.embedding_filename(language_code, i)
            def do_op(self, in_file, out_file):
                sentence_list = self.outer.get_list(in_file)
                embeddings = self.outer.embedder.get_sentence_embeddings(sentence_list)
                self.outer.embedder.save_embeddings(embeddings, out_file)

        strategy = Strategy(self)
        self.do_operation(strategy)
    
    def align_data(self, lang_code_one, lang_code_two):
        num_files = min(len(self.get_cleaned_files(lang_code_one)), len(self.get_cleaned_files(lang_code_two)))

        #create the needed directory 
        p = f"{lang_code_one}-{lang_code_two}"
        e = os.path.join(self.aligned_dir, p, "embeddings")
        s = os.path.join(self.aligned_dir, p, "sentences")

        if not os.path.exists(e):
            os.makedirs(e)
        if not os.path.exists(s):
            os.makedirs(s)

        for i in range(num_files):
            one_cleaned = self.get_language_list(lang_code_one, i)
            two_cleaned = self.get_language_list(lang_code_two, i)

            one_embedding = self.embedder.load_embeddings(self.get_embedding_file(lang_code_one, i))
            two_embedding = self.embedder.load_embeddings( self.get_embedding_file(lang_code_two, i))

            new_s, new_e = self.aligner.align_data(one_cleaned, one_embedding, two_cleaned, two_embedding)

            spath, epath = self.get_aligned_file_pair(lang_code_one, lang_code_two, i)

            #now we save them 
            self.embedder.save_embeddings(new_e, epath)
            self.save_list(new_s, spath)

    def make_line_deletions(self, line):

        if line[0] == "<":
            return "" 
        
        #convert line to lowercase 
        line = line.strip()
        
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
            if len(tokens) > 0 and tokens[0][0] == "(":
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

        if len(tokens) > 0 and tokens[-1][0] == "(":
            del tokens[-1]
        
        
        return " ".join(tokens)
    
    def save_list(self, l, p):
        l = [e + "\n" for e in l]
        with open(p, "w+", encoding="utf-8") as f:
            f.writelines(l)
    
    def get_list(self, p):
        l = []
        with open(p, "r", encoding='utf-8') as f:
            l = f.readlines()
        l = [e.strip() for e in l]
        return l
    
    def get_language_list(self, language_code, i):

        file_path = self.get_cleaned_file(language_code, i)

        if not os.path.exists(file_path):
            return None
        
        with open(file_path, "r", encoding="utf-8") as f:
            rows = []
            for line in f:
                line = line.strip()
                rows.append(line)

        return rows

    def prep_language(self, language_code):
        # self.split_files(language_code)
        # self.clean_lines(language_code)
        # self.create_embeddings(language_code)
        if language_code != "en":
            self.align_data("en", language_code)

    def prep_languages(self):
        for language_code in self.languages:
            self.prep_language(language_code)

if __name__=="__main__":
    prepper = EuroParlData(False)
    prepper.prep_language("es")


        # self.tokenize_file(language_code)
    # def get_tokenized_files(self, language_code):
    #     d = os.path.join(self.tokenized_dir, language_code)
    #     return self.get_files(d)
    
    # def get_combined_file(self, language_code):
    #     return os.path.join(self.combined_dir, f"{language_code}-combined.txt")

        # def tokenize_file(self, language_code):
    #     input_files = self.get_split_file(language_code)
    #     output_file = self.get_tokenized_file(language_code)

    #     if not self.redo_all and os.path.exists(output_file):
    #         return
        
    #     self.run_sript(self.tokenize_script, language_code, input_file, output_file)

        # def combine_files(self, language_code):
    #     #we combine the 
    #     in_fps = self.get_filepaths(language_code)
    #     output_file = self.get_combined_file(language_code)

    #     #check if it exists 
    #     if not self.redo_all and os.path.exists(output_file):
    #         return

    #     with open(output_file, "w+") as of:
    #         for in_fp in in_fps:
    #             with open(in_fp, "r") as in_f:
    #                 of.write(in_f.read())

            # self.tokenize_script = os.path.join(os.getcwd(), "europarl-data", "tools", "tokenizer.perl")
                    # self.tokenized_dir = os.path.join(self.root_dir, "tokenized-files")
