from deeppavlov import build_model, configs
import numpy as np
from utils import Recaller

model_config_path = "./config/paraphraser_bert.json"
dataset_path = "./data/paraphrase/dataset.train.txt"

model = build_model(model_config_path, download=False)
recaller = Recaller(dataset_path)

def predict(query):
    answers = recaller.retrieval_answers(query)
    if answers != []:
        querys = [query] * len(answers)
        score_matrix = model(querys, answers)
        print(score_matrix.shape)
        scores = score_matrix[:, 1] 
        index = np.argmax(scores)
        return answers[index], float(scores[index])
    else:
        return None

if __name__ == "__main__":
    print(predict("喜欢上了有对象的女孩，表白了，让她对象知道了，第二天她主动和我说话，我该怎么办"))
