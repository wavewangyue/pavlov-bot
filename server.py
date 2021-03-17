from flask import Flask
from flask import request
import json
import logging
import datetime

#from predict_paraphrase_with_recall import predict
#from predict_gobot import predict
from predict_kbqa import predict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(filename)s(line:%(lineno)d) - [%(levelname)s] %(message)s')

app = Flask(__name__)

@app.route("/talk", methods=["GET"])
def hello_world():
    
    response = {
        "result": [],
        "status": 0,
        "msg": ""
    }
    
    query = request.args.get("query", default="").strip()
    logging.info("new request: " + query)
    
    if query == "":
        response["msg"] = "query not given"
    else:
        try:
            result = predict(query)
            if result is not None:
                response["result"] = result
                response["status"] = 1
                response["msg"] = "success"
            else:
                response["msg"] = "no answer"
        except Exception as e:
            logging.error("error in server:", query, repr(e))
            response["msg"] = "server wrong"
    
    res_json = json.dumps(response, ensure_ascii=False)
    logging.info("send response: " + res_json)
    return res_json

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001, threaded=True)
