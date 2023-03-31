from .benchmarker import Benchmarker
from collections import defaultdict
import json
import os
import glob
from .testsuits import TestSuit


class BenchmarkerForNaturalInstruction(Benchmarker):

    def __init__(self, task, *args, **kwargs):
        """

        :param task (str): a string of "taskXXX" where "XXX" is the number of the task
        """
        super(BenchmarkerForNaturalInstruction, self).__init__()
        self.task = task
        self._load_dataset()

    def _load_dataset(self):
        files = glob.glob1('resources/natural-instructions/tasks/', '*.json')
        found_tasks = [x for x in files if x.startswith(self.task)]
        assert len(found_tasks) == 1, "Incorrect task name"
        task_file = found_tasks[0]

        with open(os.path.join('resources/natural-instructions/tasks/', task_file)) as f:
            data = json.load(f)

        self.task_description = '\n'.join(data['Definition'])

        for x in data['Instances']:
            input_text = x['input']
            prompt = f'{self.task_description}\n{input_text}'

            self.data.append([prompt, x['output']])

    def verbalizer(self, text):
        return text

    def metrics(self, all_trues, all_predictions):
        return {
            'bleu': self._bleu_score(all_trues, all_predictions)
        }


def test_suit_for_natural_instruction():
    t = TestSuit('natural_instruction')
    for i in range(0, 1732):
        task_name = f'task{t:03d}'
        t.register(BenchmarkerForNaturalInstruction(task=task_name, batch_size=16))
    return t
