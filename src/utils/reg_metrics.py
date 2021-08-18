from typing import *

import numpy as np
import pandas as pd


class AverageMetricTracker:
    def __init__(self, *keys, writer=None, fmt: Optional[str] = ':.6f'):
        '''
        Average metric tracker, can save multi-value
        :param keys: metrics
        :param writer:
        '''
        self.fmt = fmt
        self.writer = writer
        columns = ['total', 'counts', 'average', 'current_value']
        self._data = pd.DataFrame(np.zeros((len(keys), len(columns))), index=keys, columns=columns)
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.current_value[key] = value
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def update_multi_metrics(self, metrics: List[Dict[str, float]]):
        for metric in metrics:
            if 'n' not in metric.keys():
                metric['n'] = 1
            self.update(metric['key'], metric['value'], metric['n'])

    def avg(self, key):
        return self._data.average[key]

    def val(self, key):
        return self._data.current_value[key]

    def result(self):
        return dict(self._data.average)