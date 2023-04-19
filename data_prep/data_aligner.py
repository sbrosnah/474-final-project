import torch.nn as nn

class DataAligner:
    def __init__(self, data_prepper):
        self.cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        self.data_prepper = data_prepper

    def get_similarity(self, embedding_one, embedding_two):
        return self.cos(embedding_one, embedding_two)
