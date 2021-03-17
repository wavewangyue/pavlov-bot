fin = "tianqi.ner.txt" 
fout_train = "train1.txt"
fout_val = "valid1.txt"
fout_test = "test1.txt"

samples = []
tag2num = {}

with open(fin) as f:
    for li, line in enumerate(f.read().strip().split("\n")):
        if li % 10000 == 0: print(li)
        items = line.split("\t")
        if len(items[1].split()) > 1:
            doc = "".join(items[1].split()[1:]).strip()
        else:
            doc = items[1]
        tags = ["O"] * len(doc)
        
        for slot_ins in items[2].split("@@"):
            if slot_ins == "NULL": continue
            slot, tag = slot_ins.split("##")
            if slot not in doc: continue
            if slot == "": continue
            if tag not in ["date", "location"]: continue

            i = 0
            while i < len(doc):
                i = doc.find(slot, i)
                if i == -1: break
                if all(tags[i+j] == "O" for j in range(len(slot))):
                    for j in range(len(slot)):
                        if j == 0:
                            tags[i+j] = "B-{}".format(tag)
                        else:
                            tags[i+j] = "I-{}".format(tag)
                i += len(slot)

        samples.append((doc, tags))
        
        for tag in tags:    
            if tag not in tag2num: tag2num[tag] = 0
            tag2num[tag] += 1


p1 = int(len(samples) * 0.9)
p2 = int(len(samples) * 0.95)

samples_train = samples[:p1]
samples_val = samples[p1:p2]
samples_test = samples[p2:]

for fout, samples in [(fout_train, samples_train), (fout_val, samples_val), (fout_test, samples_test)]:
    print(fout, len(samples))
    with open(fout, "w") as w:
        for doc, tags in samples:
            for c, t in zip(doc, tags):
                w.write("\t".join([c, t]) + "\n")
            w.write("\n")

#with open("tag2num.txt", "w") as w:
#    for tag in sorted(tag2num.keys(), key=lambda tag:tag2num[tag], reverse=True):
#        w.write("\t".join([tag, str(tag2num[tag])])+"\n")


             
            
