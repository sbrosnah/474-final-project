import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
import heapq

class DataAligner:

    def __init__(self, chunk_size):
        self.cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        self.chunk_size = chunk_size
        self.filter_range = 150
        self.mat_width = 200
        self.threshold = .7

    def align_data(self, sentence_list_one, embedding_tensor_one, sentence_list_two, embedding_tensor_two):

        #check the sizes to make sure they are good 
        if len(sentence_list_one) != embedding_tensor_one.shape[0] or len(sentence_list_two) != embedding_tensor_two.shape[0]:
            print("Bad input sizes!")
            return

        new_sentences = []
        new_embeddings = []

        already_used_one = set()
        already_used_two = set()


        sim_mat = torch.matmul(embedding_tensor_one, 
                                torch.transpose(embedding_tensor_two, 0, 1))\
                                / torch.matmul(torch.norm(embedding_tensor_one, dim=1, keepdim=True), 
                                torch.transpose(torch.norm(embedding_tensor_two, dim=1, keepdim=True), 0, 1))


        similarities = []

        for i in range(embedding_tensor_one.shape[0]):
            for j in range(embedding_tensor_two.shape[0]):
                if sim_mat[i][j] < self.threshold:
                    continue 
                similarities.append((1 - sim_mat[i][j], i, j))
        
        #now we sort the sentences by the similarity
        heapq.heapify(similarities)

        while len(similarities) > 0:
            _, i, j = heapq.heappop(similarities)

            if i in already_used_one or j in already_used_two:
                continue 

            already_used_one.add(i)
            already_used_two.add(j)

            new_sentences.append(sentence_list_one[i])
            new_sentences.append(sentence_list_two[j])
            new_embeddings.append(embedding_tensor_one[i])
            new_embeddings.append(embedding_tensor_two[j])
            
            
        return new_sentences, torch.stack(new_embeddings, dim=0)