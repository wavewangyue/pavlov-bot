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

@register('my_query_generator')
class MyQueryGenerator(Component, Serializable):
    
    def __init__(self, max_entity_num=5, max_relation_num=5, **kwargs):
        super().__init__(**kwargs)
        self.max_entity_num = max_entity_num
        self.max_relation_num = max_relation_num

    @overrides
    def __call__(self, templates_batch, entities_batch, relations_batch, *args, **kwargs):
        sqls_batch = []
        for template, entities, relations in zip(templates_batch, entities_batch, relations_batch):
            # filling entities
            for i in range(self.max_entity_num):
                if "@Ent{}".format(i+1) in template:
                    entity = entities[i] if len(entities) > i and entities[i] is not None else "e.missing"
                    template = template.replace("@Ent{}".format(i+1), "\'{}\'".format(entity))
            if "@Ent" in template:
                entity = entities[0] if len(entities) > 0 and entities[0] is not None else "e.missing"
                template = template.replace("@Ent", "\'{}\'".format(entity))
            # filling relations
            relations = relations.split()
            for i in range(self.max_relation_num):
                if "@Rel{}".format(i+1) in template:
                    relation = relations[i] if len(relations) > i and relations[i] is not None else "r.missing"
                    template = template.replace("@Rel{}".format(i+1), "\'{}\'".format(relation))
            if "@Rel" in template:
                relation = relations[0] if len(relations) > 0 and relations[0] is not None else "r.missing"
                template = template.replace("@Rel", "\'{}\'".format(relation))
            # sql
            sqls_batch.append(template)    
        return sqls_batch
                    
    def load(self, *args, **kwargs):
        pass
         
    def save(self, *args, **kwargs):
        pass

