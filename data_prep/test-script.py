from bert_embedder import BertEmbedder
from europarl_data import EuroParlData

batch_size = 2
save_multiple = 10
clean_multiple = 5
start_where_left = False

embedder = BertEmbedder(batch_size, save_multiple, clean_multiple, start_where_left)
data = EuroParlData(False)

en_list = data.get_language_list('en')[:150]
print(len(en_list))
emb = embedder.get_sentence_embeddings(en_list)

print(emb.shape)
print("raw", embedder.get_sentence_embeddings(en_list[-2:]))
# print("raw", embedder.get_sentence_embeddings(en_list[-2:]))
print("from saved", emb[-2:])

print("raw:", embedder.get_sentence_embeddings(en_list[37:39]))
print("from saved:", emb[37:39])