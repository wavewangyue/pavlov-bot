import random
import collections
from xml.dom import minidom

# def word_segment(sentence):
#     return [word.strip() for word in jieba.cut(sentence) if word.strip() != ""]

def process_train(fin, fout, neg_samples=10):
    lines = []
    pairs = []
    with open(fin, "r", encoding="utf-8") as f:
        lines = f.read().strip().split("\n")
    print("samples:", len(lines))
    for i in range(len(lines)):
        items = lines[i].split("\t")
        sid1, q1, a1 = items[0], items[1], items[2]
        pair = collections.OrderedDict()
        pair["id"] = len(pairs) 
        pair["id_1"] = "{}_q".format(sid1) 
        pair["id_2"] = "{}_a".format(sid1)
        pair["text_1"] = q1
        pair["text_2"] = a1
        pair["class"] = 1
        pairs.append(pair)
        for _ in range(neg_samples):
            j = i
            while j == i:
    	        j = random.randint(0, len(lines)-1)
            items = lines[j].split("\t")
            sid2, q2, a2 = items[0], items[1], items[2]
            pair = collections.OrderedDict()
            pair["id"] = len(pairs) 
            pair["id_1"] = "{}_q".format(sid1) 
            pair["id_2"] = "{}_a".format(sid2)
            pair["text_1"] = q1
            pair["text_2"] = a2
            pair["class"] = -1
            pairs.append(pair)
    print("pairs:", len(pairs))
    # write to xml
    dom = minidom.getDOMImplementation().createDocument(None, None, None) # namespaceURI,qualifiedName,doctype
    root = dom.createElement("data")
    corpus = dom.createElement("corpus")
    for pair in pairs:
        paraphrase = dom.createElement("paraphrase")
        for k, v in pair.items():
            value = dom.createElement("value")
            value.setAttribute("name", k)
            value.appendChild(dom.createTextNode(str(v)))
            paraphrase.appendChild(value)
        corpus.appendChild(paraphrase)
    root.appendChild(corpus)
    dom.appendChild(root)
    with open(fout, 'w', encoding='utf-8') as w:
        dom.writexml(w, addindent='  ', newl='\n',encoding='utf-8')


def process_test(fin, fout, neg_samples):
    samples = []
    with open(fin) as f:
        for line in f.read().strip().split("\n"):
            items = line.split("\t")
            sid, q, a = items[0:3]
            samples.append((sid, q, a))
    print("samples:", len(samples))
    with open(fout, "w") as w:
        for sample in samples:
            sid1, q1, a1 = sample
            w.write("\t".join(["{}_q".format(sid1), q1, "{}_a".format(sid1), a1, "1"]) + "\n")
            for _ in range(neg_samples):
                sample2 = sample
                while sample2 == sample:
                    sample2 = random.choice(samples)
                sid2, q2, a2 = sample2
                w.write("\t".join(["{}_q".format(sid1), q1, "{}_a".format(sid2), a2, "0"]) + "\n")


#process_train("./dataset.train.txt", "./paraphrases.xml", neg_samples=9)
#process_train("./dataset.test.txt", "./paraphrases_gold.xml", neg_samples=5)
process_test("./dataset.test.txt", "./paraphrases_test.txt", neg_samples=9)


