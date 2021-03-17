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
from logging import getLogger

from rapidfuzz import process
from overrides import overrides

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.utils import download
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.serializable import Serializable

log = getLogger(__name__)


@register('my_slotfilling')
class MySlotFilling(Component, Serializable):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @overrides
    def __call__(self, tokens_batch, tags_batch, *args, **kwargs):
        slots = [{}] * len(tokens_batch)
        m = [i for i, v in enumerate(tokens_batch) if v]
        if m:
            tags_batch = [tags_batch[i] for i in m]
            tokens_batch = [tokens_batch[i] for i in m]
            for i, tokens, tags in zip(m, tokens_batch, tags_batch):
                slots[i] = self.predict_slots(tokens, tags)
        return slots

    def predict_slots(self, tokens, tags):
        # For utterance extract named entities and perform normalization for slot filling

        entities, slots = self._chunk_finder(tokens, tags)
        slot_values = {}
        for entity, slot in zip(entities, slots):
            slot_values[slot] = entity.replace(' ', '')
        return slot_values

    @staticmethod
    def _chunk_finder(tokens, tags):
        # For BIO labeled sequence of tags extract all named entities form tokens
        prev_tag = ''
        chunk_tokens = []
        entities = []
        slots = []
        for token, tag in zip(tokens, tags):
            curent_tag = tag.split('-')[-1].strip()
            current_prefix = tag.split('-')[0]
            if tag.startswith('B-'):
                if len(chunk_tokens) > 0:
                    entities.append(' '.join(chunk_tokens))
                    slots.append(prev_tag)
                    chunk_tokens = []
                chunk_tokens.append(token)
            if current_prefix == 'I':
                if curent_tag != prev_tag:
                    if len(chunk_tokens) > 0:
                        entities.append(' '.join(chunk_tokens))
                        slots.append(prev_tag)
                        chunk_tokens = []
                else:
                    chunk_tokens.append(token)
            if current_prefix == 'O':
                if len(chunk_tokens) > 0:
                    entities.append(' '.join(chunk_tokens))
                    slots.append(prev_tag)
                    chunk_tokens = []
            prev_tag = curent_tag
        if len(chunk_tokens) > 0:
            entities.append(' '.join(chunk_tokens))
            slots.append(prev_tag)
        return entities, slots

    def save(self, *args, **kwargs):
        pass

    def load(self, *args, **kwargs):
        pass
