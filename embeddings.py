import torch
import esm
import pandas as pd
from Bio import SeqIO

MODEL_NAME = "esm2_t6_8M_UR50D"
LAYER = 6

print("Loading model...")
model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
model.eval()

batch_converter = alphabet.get_batch_converter()

records = list(SeqIO.parse("mutants.fasta", "fasta"))
data = [(record.id, str(record.seq)) for record in records]

labels, sequences, tokens = batch_converter(data)

print("Generating embeddings...")
with torch.no_grad():
    results = model(tokens, repr_layers=[LAYER])

token_embeddings = results["representations"][LAYER]

vectors = []
for i, label in enumerate(labels):
    seq_len = len(sequences[i])
    residue_embeddings = token_embeddings[i, 1:seq_len + 1]
    mean_embedding = residue_embeddings.mean(0).cpu().numpy()

    row = {"id": label}
    for j, value in enumerate(mean_embedding):
        row[f"e{j}"] = float(value)
    vectors.append(row)

df = pd.DataFrame(vectors)
df.to_csv("embeddings.csv", index=False)

print("Saved embeddings.csv")