import random
import json

fin = "./data/EFA_Dataset_v20200314_latest.txt"
fout1 = "./data/dataset.train.txt"
fout2 = "./data/dataset.test.txt"

lines_w = []
with open(fin, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        ins = json.loads(line)
        q = ins["title"].lstrip("男女,，").replace("\n", " ").replace("\t", " ").strip()
        label1 = ins["label"]["s1"]
        label2 = ins["label"]["s2"]
        label3 = ins["label"]["s3"]
        best_ans = ""
        max_score = -1
        for chat in ins["chats"]:
            if chat["sender"] != "audience": continue
            ans = chat["value"].replace("\n", " ").replace("\t", " ").strip()
            score = 0
            if "label" in chat:
                if chat["label"]["negative"]: 
                    score = -1
                else:
                    if chat["label"]["knowledge"]:
                        score += 10
                    if chat["label"]["question"]:
                        score += 1
            if score > max_score:
                best_ans = ans
                max_score = score
        if q != "" and best_ans != "":
            lines_w.append("\t".join([
                "sample_{}".format(str(i)),
                q.replace("\n", " ").replace("\t", " "),
                best_ans.replace("\n", " ").replace("\t", " "),
                label1,
		label2,
		label3
            ]))

random.shuffle(lines_w)
print("samples:", len(lines_w))
p = int(len(lines_w) * 0.9)
lines_train = lines_w[:p]
lines_test = lines_w[p:]
print("train samples:", len(lines_train))
print("test samples:", len(lines_test))

with open(fout1, "w", encoding="utf-8") as w:
    for line in lines_train:
        w.write(line + "\n")

with open(fout2, "w", encoding="utf-8") as w:
    for line in lines_test:
        w.write(line + "\n")
            
