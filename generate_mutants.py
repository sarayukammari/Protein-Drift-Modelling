from Bio import SeqIO

wt = str(next(SeqIO.parse("wt.fasta", "fasta")).seq)

def mutate(seq, mutation):
    ref = mutation[0]
    pos = int(mutation[1:-1]) - 1
    alt = mutation[-1]

    if seq[pos] != ref:
        raise ValueError(
            f"Reference mismatch for {mutation}: expected {ref}, found {seq[pos]}"
        )

    return seq[:pos] + alt + seq[pos + 1:]

mutants = {}

with open("mutations.txt") as f:
    for line in f:
        mutation = line.strip()
        if mutation:
            mutants[mutation] = mutate(wt, mutation)

with open("mutants.fasta", "w") as f:
    f.write(">WT\n" + wt + "\n")
    for name, seq in mutants.items():
        f.write(f">{name}\n{seq}\n")

print("Created mutants.fasta")


