from deeppavlov import build_model, configs
import sqlite3

model_config_path = "./config/kbqa.json"
database_path = "./data/kbqa/stars.db"
database_table_name = "BAIKE_STAR_TRIPLES"

model = build_model(model_config_path, download=True)
conn = sqlite3.connect(database_path, check_same_thread=False)
cursor = conn.cursor()

def generate_response(mention, relation, template, result):
    m = mention[0]
    r = "和".join(relation)
    if "Count" in template:
        s = "{}的{}有{}个".format(m, r, result[0][0])
    elif len(result) > 1:
        s = "{}的{}有{}".format(m, r, "、".join(row[0] for row in result))
    else:
        s = "{}的{}是{}".format(m, r, result[0][0])
    return s
    
def predict(query):
    mentions, entities, relations, templates, sqls= model([" ".join(list(query))])
    mention, entity, relation, template, sql = mentions[0], entities[0], relations[0], templates[0], sqls[0]
    relation = relation.split()
    sql = sql.replace("@Table", database_table_name)
    try:
        result = list(cursor.execute(sql))
        if result != []:
            response = generate_response(mention, relation, template, result)
        else:
            response = "抱歉，没有在知识库中找到结果"
    except Exception as e:
        print(repr(e))
        result = []
        response = "抱歉，没有在知识库中找到结果"

    return {
        "mention": mention,
        "entity": entity,
        "relation": relation,
        "sql": sql,
        "result": result,
        "response": response
    }

if __name__ == "__main__":
    print(predict("赵本山的孩子是谁"))
    print(predict("赵本山的孩子有几个"))
    print(predict("赵本山的出生地是哪"))
    print(predict("赵本山的学生都有谁"))

