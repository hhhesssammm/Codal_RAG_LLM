from copy import deepcopy
from types import SimpleNamespace
import numpy as np



class FakeLLM:
    def __init__(self, response=None):
        self.response = response
        self.calls = []

    def generate_json(self, system_prompt, user_prompt):
        self.calls.append((system_prompt, user_prompt))
        if self.response is not None:
            return deepcopy(self.response)
        query = user_prompt.split('سؤال:\n', 1)[1].split('\nشواهد بازیابی‌شده:', 1)[0].strip()
        claim = {'claim': 'فروش شرکت کاهش یافته است.', 'evidence_ids': ['E1'],
                 'basis': 'شاهد بازیابی‌شده', 'confidence': 0.9}
        return {'query': query, 'explicit': [claim], 'implicit': [], 'latent': [],
                'insufficient_evidence': False, 'notes': 'Mocked test output'}


class FakeSentenceModel:
    max_seq_length = 512

    def __init__(self):
        self.calls = []
        self.tokenizer = SimpleNamespace(encode=lambda text, **kwargs: text.split() + ['<s>', '</s>'])

    def get_sentence_embedding_dimension(self):
        return 3

    def encode(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        return np.tile([1., 0., 0.], (len(texts), 1))
