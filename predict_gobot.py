from deeppavlov import build_model, configs
import numpy as np
import requests
import datetime

from predict_paraphrase_with_recall import predict as predict_psycho_faq 
from predict_kbqa import predict as predict_stars_kbqa

model_config_path = "./config/gobot.json"

model = build_model(model_config_path, download=False)

def get_weather_api(location, date):
    # location:北京市 date:明天
    # 处理日期格式
    if date == "今天":
        date_p = datetime.datetime.now().strftime("%Y%m%d")
    elif date == "明天":
        date_p = (datetime.datetime.now()+datetime.timedelta(days=1)).strftime("%Y%m%d")
    elif date == "后天":
        date_p = (datetime.datetime.now()++datetime.timedelta(days=2)).strftime("%Y%m%d")
    else:
        return "无法识别的日期：{}".format(date)
    # 调用api
    url = "http://route.showapi.com/9-11"
    params = {
        "showapi_appid": "551929",
        "showapi_sign": "60beaa96d9ca4b31abe8adb27ca117d9",
        "area": location,
        "date": date_p,
        "need3HourForcast": "0"
    }
    res = requests.get(url, params=params).json()
    if res["showapi_res_code"] != 0:
        return "API错误：{}".format(res["showapi_res_error"])
    elif res["showapi_res_body"]["ret_code"] != 0:
        return "API错误：{}".format(res["showapi_res_body"]["remark"])
    # 生成回复文本
    for f in ["f1", "f2", "f3"]:
        if f in res["showapi_res_body"]:  
            response = "{}{}天气{}，白天气温{}摄氏度，夜晚气温{}摄氏度，风向{}{}，紫外线强度{}，降水概率{}。{}".format(
                location, date, 
                res["showapi_res_body"][f]["day_weather"],
                res["showapi_res_body"][f]["day_air_temperature"], res["showapi_res_body"][f]["night_air_temperature"],
                res["showapi_res_body"][f]["day_wind_direction"], res["showapi_res_body"][f]["day_wind_power"],
                res["showapi_res_body"][f]["ziwaixian"], res["showapi_res_body"][f]["jiangshui"],
                res["showapi_res_body"][f]["index"]["xq"]["desc"]
            )
            return response
    return "API错误：未获取到天气结果"


def predict(query):
    query_p = " ".join(list(query.replace(" ", "")))
    actions, responses, slots, intents = model([query_p])
    action, response, slot, intent = actions[0], responses[0], slots[0], intents[0]
    result = {"action": action, "response": response, "slot": slot, "intent": intent}
    # action处理
    if action == "act_weather_inform":
        location, date = response.split(" ")
        response = get_weather_api(location, date)
        result["response"] = response
    elif action == "act_psycho_faq":
        response, score = predict_psycho_faq(query)
        result["response"] = response
    elif action == "act_stars_kbqa":
        kbqa_result = predict_stars_kbqa(query)
        result["response"] = kbqa_result["response"]
        kbqa_result.pop("response")
        result["kbqa_result"] = kbqa_result
    # return
    return result


if __name__ == "__main__":
    #print(predict("北京市明天天气怎么样"))
    print(predict("赵本山的孩子是谁"))
