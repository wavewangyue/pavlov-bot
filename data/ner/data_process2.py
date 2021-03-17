fin = "train_finetuning_total.tsv"
fout_train = "train2.txt"
fout_val = "valid2.txt"
fout_test = "test2.txt"

samples = []
with open(fin) as f:
    samples = f.read().strip().split("\n\n")

samples_p = []
for sample in samples:
    lines = sample.strip().split("\n")
    chars = []
    tags = []
    for line in sample.strip().split("\n"):
        items = line.split()
        ch = items[0]
        tag = items[2]
        if not any(w in tag for w in ["-date", "-location"]):
            tag = "O"
        chars.append(ch)
        tags.append(tag)
        
    if all(tag == "O" for tag in tags):
        continue

    samples_p.append((chars, tags))

print(len(samples_p), "/", len(samples))             

p1 = int(len(samples_p) * 0.9)
p2 = int(len(samples_p) * 0.95)

samples_train = samples_p[:p1]
samples_val = samples_p[p1:p2]
samples_test = samples_p[p2:]

for fout, samples in [(fout_train, samples_train), (fout_val, samples_val), (fout_test, samples_test)]:
    print(fout, len(samples))
    with open(fout, "w") as w:
        for doc, tags in samples:
            for c, t in zip(doc, tags):
                w.write("\t".join([c, t]) + "\n")
            w.write("\n")

