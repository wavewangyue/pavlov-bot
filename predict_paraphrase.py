from deeppavlov import build_model, configs
import numpy as np

model_config_path = "./config/paraphraser_bert.json"
fin = "./data/paraphrase/paraphrases_test.txt"

model = build_model(model_config_path, download=False)

sid2sample = {}
with open(fin) as f:
    for line in f.read().strip().split("\n"):
        sid_q, q, sid_a, a, label = line.split("\t")
        sid = sid_q
        if sid not in sid2sample: sid2sample[sid] = {"q": q, "as": [], "gold": 0}
        if label == "1": sid2sample[sid]["gold"] = len(sid2sample[sid]["as"])
        sid2sample[sid]["as"].append(a)

samples = sid2sample.values()
hits_1 = 0
hits_3 = 0
hits_5 = 0
for i, sample in enumerate(samples):
    if i % 100 == 0: print(i, "/", len(samples))
    answers = sample["as"]
    querys = [sample["q"]] * len(answers)
    score_matrix = model(querys, answers)
    scores = score_matrix[:, 1] 
    indexes_sorted = np.argsort(scores)[::-1]
    index_gold = sample["gold"]
    if index_gold in indexes_sorted[:1]: hits_1 += 1
    if index_gold in indexes_sorted[:3]: hits_3 += 1
    if index_gold in indexes_sorted[:5]: hits_5 += 1

print("p_1:", hits_1*100 / len(samples), hits_1, len(samples))
print("p_3:", hits_3*100 / len(samples), hits_3, len(samples))
print("p_5:", hits_5*100 / len(samples), hits_5, len(samples))

