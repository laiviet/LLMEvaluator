from .benchmarker import Benchmarker
from collections import defaultdict
import json

from .testsuits import TestSuit


class BenchmarkerForXNLI(Benchmarker):
    all_languages = 'en,es'.split(',')

    def __init__(self, language: str = 'all', *args, **kwargs):
        """

        :param languages (str): a single language code
                                if languages = "all", all available languages will be selected
        """
        assert type(language) == str

        if language == 'all':
            self.languages = BenchmarkerForXNLI.all_languages
            self.name = 'all'
        else:
            assert language in BenchmarkerForXNLI.all_languages
            self.languages = [language]
            self.name = language
        super(BenchmarkerForXNLI, self).__init__(*args, **kwargs)

    def _set_task_description(self):
        return 'Please identify whether the premise entails or contradicts the hypothesis' \
               ' in the following premise and hypothesis. ' \
               'The answer should be a single word: "entailment", "contradiction", or "neutral".'

    def _load_dataset(self):
        file = 'resources/XNLI-1.0/xnli.dev.jsonl'
        with open(file) as f:
            lines = f.readlines()
        data = []
        for line in lines:
            sample = json.loads(line)
            lang = sample['language']
            if lang in self.languages:
                premise = sample['sentence1']
                hypothesis = sample['sentence2']
                gold_label = sample['gold_label']
                input = f'{self.task_description}\nPremise: {premise}\nHypothesis: {hypothesis}\nThe premise and the hypothesis is '
                data.append([input, gold_label])
        print('-' * 80)
        print(f'Loaded test suit for {self.name}', len(data), ' samples')

        return data

    def verbalizer(self, text):
        text = text.lower()
        is_a = 'entail' in text and 'not entail' not in text
        is_b = 'contradict' in text and 'not contradict' not in text
        is_c = 'neutral' in text and 'not neutral' not in text

        if is_a and not is_b and not is_c:
            return 'entailment'
        if is_b and not is_a and not is_c:
            return 'contradiction'
        if is_c and not is_a and not is_b:
            return 'neutral'
        return 'No answer'

    def metrics(self, all_trues, all_predictions):
        return {
            'accuracy': self._accuracy_score(all_trues, all_predictions)
        }


def testsuit_for_XNLI():
    t = TestSuit('XNLI')
    for lang in BenchmarkerForXNLI.all_languages:
        t.register(BenchmarkerForXNLI(language=lang, batch_size=16))
    return t
