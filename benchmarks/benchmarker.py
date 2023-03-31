from sklearn.metrics import accuracy_score
import tqdm


class Benchmarker():

    def __init__(self, batch_size=2):
        super(Benchmarker, self).__init__()
        self.batch_size = batch_size
        self.task_description = ''
        self.data = []

    def __iter__(self):
        total = len(self.data) // self.batch_size
        for i in tqdm.tqdm(range(0, len(self.data), self.batch_size), total=total):
            data = self.data[i:i + self.batch_size]
            yield [x[0] for x in data], [x[1] for x in data]

    def _set_task_description(self):
        return ''

    def _load_dataset(self):
        raise NotImplemented

    def _accuracy_score(self, all_trues, all_predictions):
        acc = accuracy_score(all_trues, all_predictions) * 100.0
        return acc

    def _bleu_score(self, all_trues, all_predictions):
        return 0.0
