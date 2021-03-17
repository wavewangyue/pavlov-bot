from deeppavlov import build_model, configs
import sqlite3

model = build_model("./config/kbqa.json", download=True)

query = "赵本山的孩子是谁"
print(model([" ".join(list(query))]))


