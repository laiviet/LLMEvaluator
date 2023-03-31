import json


class TestSuit():

    def __init__(self, name: str):
        super(TestSuit, self).__init__()
        self.name = name
        self._benchmark = []
        self.reports = {}

    def register(self, benchmark):
        self._benchmark.append(benchmark)

    def test(self, model):
        for benchmark in self._benchmark:
            all_trues = []
            all_predictions = []
            for prompts, labels in benchmark:
                answers = model.generate(prompts)
                print(answers)
                answers = [benchmark.verbalizer(x) for x in answers]
                print(answers)
                print(labels)
                all_trues.extend(labels)
                all_predictions.extend(answers)

            metrics = benchmark.metrics(all_trues, all_predictions)
            self.reports[benchmark.name] = metrics

    def print_report(self):
        print(f'Final report for {self.name}')
        for bm, metrics in self.reports.items():
            for metric_name, value in metrics.items():
                print(f'{bm} > {metric_name}: {value:.2f}')

    def report(self):
        return self.reports

    def json_report(self):
        return json.dumps(self.reports, indent=2)
