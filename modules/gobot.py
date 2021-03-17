# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import random
import datetime
from logging import getLogger

from overrides import overrides

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.serializable import Serializable

log = getLogger(__name__)

@register('my_gobot')
class MyGoBot(Component, Serializable):
    
    def __init__(self, max_patience_keep_a_session=None, max_time_keep_a_session=None, **kwargs):
        super().__init__(**kwargs)
        self.max_patience_keep_a_session = max_patience_keep_a_session
        self.max_time_keep_a_session = max_time_keep_a_session
        self.intents = []
        self.action2responses = {}
        self.intent2patience = {}
        self.intent2updatetime = {}
        self.intent2state = {}
        self.load()

    @overrides
    def __call__(self, slots_batch, intents_batch, *args, **kwargs):
        actions = [] * len(slots_batch)
        responses = [] * len(slots_batch)
        for slot, intent in zip(slots_batch, intents_batch):
            action = self.decide_action(intent, slot) # 根据slot和intent获取动作
            state = self.intent2state[intent] if intent in self.intent2state else None
            response = self.get_response(action, state) # 根据act获取回复
            actions.append(action)
            responses.append(response)
            self.clear_state(intent) # 根据规则清理对话状态
        return actions, responses

    # 根据 slot+intent+state 决定采取什么动作
    def decide_action(self, intent, slot):  
        if intent not in self.intents:
            return "err_intent_undefined:{}".format(intent)
        elif intent == "Chat":
            return "act_chat_faq"
        elif intent == "Psycho":
            return "act_psycho_faq"
        elif intent == "Stars":
            return "act_stars_kbqa"
        elif intent == "Weather":
            if intent in self.intent2state:
                state = self.intent2state[intent]
                for slot_k, slot_v in slot.items():
                    if slot_k not in state:
                        return "err_slot_undefined:{}/{}".format(intent, slot_k)
                    else:
                        state[slot_k] = slot_v
                if (state["location"] is None) and (state["date"] is None):
                    return "act_weather_req_loc_and_dt"
                elif state["location"] is None:
                    return "act_weather_req_loc"
                elif state["date"] is None:
                    return "act_weather_req_dt"
                else:
                    return "act_weather_inform"
        # 如果未命中以上的各种规则，表示有遗漏情况
        return "err_intent_unprocessed:{}".format(intent)

    # 根据action获取回复
    def get_response(self, action, state):
        if action in self.action2responses:
            response = random.choice(self.action2responses[action])
            if state is not None:
                for slot_k, slot_v in state.items():
                    if slot_v is not None:
                        response = response.replace("##{}##".format(slot_k), slot_v)
            return response
        else:
            return ""

    # 清理对话状态
    def clear_state(self, intent):
        for intent_keep in self.intent2patience:
            if intent_keep == intent:
                self.intent2patience[intent_keep] = 0
                self.intent2updatetime[intent_keep] = datetime.datetime.now()
                # 如果该意图下已经完成了一组交互，即所有槽值都已经填满，则清理状态
                if all(slot_v is not None for slot_k, slot_v in self.intent2state[intent_keep].items()):
                    for slot_k in self.intent2state[intent_keep]:
                        self.intent2state[intent_keep][slot_k] = None
            else:
                # 如果跳出某个意图时间过久或过多轮次，则清理状态
                self.intent2patience[intent_keep] += 1
                time_passed = int((datetime.datetime.now() - self.intent2updatetime[intent_keep]).total_seconds())
                if ((self.max_patience_keep_a_session and self.intent2patience[intent_keep] >= self.max_patience_keep_a_session) or 
                    (self.max_time_keep_a_session and time_passed >= self.max_time_keep_a_session)):
                    for slot_k in self.intent2state[intent_keep]:
                        self.intent2state[intent_keep][slot_k] = None
                    
    # 加载json文件
    def load(self, *args, **kwargs):
        with open(self.load_path, "r", encoding="utf8") as f:
            config = json.load(f)
            for intent in config["intents"]:
                self.intents.append(intent["name"])
                if "slots" in intent:
                    self.intent2patience[intent["name"]] = 0
                    self.intent2updatetime[intent["name"]] = datetime.datetime.now()
                    self.intent2state[intent["name"]] = {slot: None for slot in intent["slots"]}
            for action in config["actions"]:
                if "response" in action:
                    self.action2responses[action["name"]] = action["response"]
    
    def save(self, *args, **kwargs):
        pass

