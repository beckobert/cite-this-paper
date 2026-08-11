# test_embedding.py

from FlagEmbedding import FlagAutoModel
import torch

print("GPU:", torch.cuda.get_device_name(0))

model = FlagAutoModel.from_finetuned(
    "BAAI/bge-m3",
    use_fp16=True,
)

texts = [
    "Participatory design increases users' sense of ownership.",
    "Users involved in the design process reported greater perceived ownership.",
    "The experiment measured the thermal conductivity of aluminium.",
]

embeddings = model.encode(texts)

print(embeddings.keys())
print("Shape:", embeddings['dense_vecs'].shape)
print()
print("Similarities:")
print(embeddings['dense_vecs'] @ embeddings['dense_vecs'].T)
