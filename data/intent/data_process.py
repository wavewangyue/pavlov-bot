import random
import csv
import re

fin = "total5_label.txt"
fout_train = "train.csv"
fout_test = "test.csv"

samples = []

with open(fin) as f:
    for line in f.read().strip().split("\n"):
        if len(line.split("\t")) != 2:
            print(line)
        tag, doc = line.split("\t")
        doc = doc.strip()
        if doc == "": continue
        if tag == "Other": continue
        if tag == "Psycho":
            for subdoc in re.split('，|。|！|？|\s', doc):
                subdoc = subdoc.strip()
                if subdoc != "" and len(subdoc) >= 6:
                    samples.append((subdoc, tag))
        samples.append((doc, tag))

random.shuffle(samples)
p = int(len(samples) * 0.95)
samples_train = samples[:p]
samples_test = samples[p:]

for fout, samples in [(fout_train, samples_train), (fout_test, samples_test)]:
    print(fout, len(samples))
    with open(fout, "w") as w:
        writer = csv.writer(w)
        writer.writerow(["text", "label"])
        for doc, tag in samples:
            writer.writerow([doc, tag])
