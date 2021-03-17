import gensim
import jieba.posseg as pseg
from gensim.summarization import bm25


class Recaller():
    
    def __init__(self, dataset_path):
        corpus = []
        sids_p = []
        sid2qa = {}
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f.read().strip().split("\n"):
                items = line.split("\t")
                sid, q, a = items[0], items[1], items[2]
                words = self.word_segment(q)
                if words != []:
                    corpus.append(words)
                    sids_p.append(sid)
                    sid2qa[sid] = (q, a)
        sids = sids_p
        print("build bm25 model...")
        model = bm25.BM25(corpus)
        print("build bm25 model with corpus:", len(corpus))
        self.sids = sids
        self.model = model
        self.sid2qa = sid2qa

    def word_segment(self, sentence):
        stop_words = []
        words = []
        for word, flag in pseg.cut(sentence):
            word = word.strip()
            if word != "" and flag not in ['x', 'c', 'u', 'p', 'uj', 'r']:
                words.append(word)
        return words

    def retrieval_answers(self, doc, k=50):
        words = self.word_segment(doc)
        if words != []:
            scores = self.model.get_scores(words)
            sid2score = dict(zip(self.sids, scores))
            sids_topK = sorted(self.sids, key=lambda sid:sid2score[sid], reverse=True)[:k]
            return [self.sid2qa[sid][1] for sid in sids_topK]
        else:
            return []



