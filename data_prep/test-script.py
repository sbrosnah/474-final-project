from bert_embedder import BertEmbedder
from europarl_data import EuroParlData
from data_aligner import DataAligner

batch_size = 24
save_multiple = 10
clean_multiple = 5
start_where_left = False

embedder = BertEmbedder(batch_size, save_multiple, clean_multiple, start_where_left)
data = EuroParlData(False)
aligner = DataAligner(5)

# en_list = data.get_language_list("en")[:1000]
# es_list = data.get_language_list("es")[:1000]

data.prep_language("en")
data.prep_language("es")

# en_embeddings = embedder.get_sentence_embeddings(en_list)
# es_embeddings = embedder.get_sentence_embeddings(es_list)

# embedder.save_embeddings(en_embeddings, data.get_embedding_file("en"))
# embedder.save_embeddings(es_embeddings, data.get_embedding_file("es"))

# en_embeddings = embedder.load_embeddings(data.get_embedding_file("en"))
# es_embeddings = embedder.load_embeddings(data.get_embedding_file("es"))

# s1, s2, e1, e2= aligner.align_data(en_list, en_embeddings, es_list, es_embeddings)

# for senOne, senTwo in zip(s1[400:430], s2[400:430]):
#     print("English:", senOne)
#     print("Spanish:", senTwo, "\n")

# print("English length:", len(s1))
# print("Spanish length:", len(s2))

# for senOne, senTwo in zip(en_list, es_list):
#     print("English:", senOne)
#     print("Spanish:", senTwo, "\n")
