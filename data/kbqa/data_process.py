import json
import csv

fin = "stars_kbqa_dataset.txt"
fouts_train = ("ner/train.txt", "relation_cls/train.csv", "template_cls/train.csv")
fouts_valid = ("ner/valid.txt", "relation_cls/valid.csv", "template_cls/valid.csv")
fouts_test = ("ner/test.txt", "relation_cls/test.csv", "template_cls/test.csv")

samples = []
with open(fin) as f:
    for line in f.read().strip().split("\n"):
        samples.append(json.loads(line))
p1 = int(len(samples) * 0.9)
p2 = int(len(samples) * 0.95)
samples_train = samples[:p1]
samples_valid = samples[p1:p2]
samples_test = samples[p2:]

for samples, (fout_ner, fout_rc, fout_tc) in [(samples_train, fouts_train), (samples_valid, fouts_valid), (samples_test, fouts_test)]:
    samples_ner = []
    samples_rc = []
    samples_tc = []
    for sample in samples:
        query = sample["query"]
        ent_name = list(sample["entities"].values())[0]
        # ner
        chars = list(query)
        bios = ["O"] * len(chars)
        i = query.index(ent_name)
        for k in range(len(ent_name)):
            bios[i+k] = "B-PERSON" if k == 0 else "I-PERSON"
        samples_ner.append((chars, bios))
        # relation cls
        relations = list(sample["relations"].values())
        samples_rc.append((query, relations))
        # template cls
        template = sample["sql"]
        samples_tc.append((query, template))
        
    with open(fout_ner, "w") as w:
        print(fout_ner, len(samples_ner))
        for chars, bios in samples_ner:
            for char, bio in zip(chars, bios):
                w.write("\t".join([char, bio]) + "\n")
            w.write("\n")

    with open(fout_rc, "w") as w:
        print(fout_rc, len(samples_rc))
        writer = csv.writer(w)
        writer.writerow(["text", "label"])
        for query, relations in samples_rc:
            writer.writerow([query, " ".join(relations)])

    with open(fout_tc, "w") as w:
        print(fout_tc, len(samples_tc))
        writer = csv.writer(w)
        writer.writerow(["text", "label"])
        for query, template in samples_tc:
            writer.writerow([query, template])


