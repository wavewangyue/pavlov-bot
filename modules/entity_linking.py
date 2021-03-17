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

@register('my_entity_linking')
class MyEntityLinking(Component, Serializable):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mention2entity = {}
        self.load()

    @overrides
    def __call__(self, mentions_batch, *args, **kwargs):
        entities_batch = []
        for mentions in mentions_batch:
            entities = [self.mention2entity[mention] if mention in self.mention2entity else None for mention in mentions]
            entities_batch.append(entities)
        return entities_batch
                    
    # 加载mention2entity映射表
    def load(self, *args, **kwargs):
        with open(self.load_path, "r", encoding="utf8") as f:
            self.mention2entity = dict([line.split("\t") for line in f.read().strip().split("\n")])
    
    def save(self, *args, **kwargs):
        pass

