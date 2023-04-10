from copy import copy

from ignite.metrics import Metric
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced


class MultiChannelMetric(Metric):
    def __init__(self, metric: Metric, channels):
        self._metric = [copy(metric) for i in range(channels)]
        self._num_examples = channels
        super(MultiChannelMetric, self).__init__()

    @reinit__is_reduced
    def reset(self):
        self._num_examples = 0
        for metric in self._metric:
            metric.reset()
        super(MultiChannelMetric, self).reset()

    @reinit__is_reduced
    def update(self, output):
        y_pred, y = output[0].detach(), output[1].detach()
        for i, metric in enumerate(self._metric):
            metric.update((y_pred[:, i, :, :].unsqueeze(1), y[:, i, :, :].unsqueeze(1)))
        self._num_examples += y.shape[0]

    @sync_all_reduce("_num_examples")
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError(
                "MultiChannelMetric must have at least one example before it can be computed."
            )

        results = []
        for metric in self._metric:
            result = metric.compute()
            if type(result) != float:
                result = result.item()
            results.append(result)

        return results


def get_multichannel_metric_names(metric_keys, channel_names):
    metric_names = []
    for key in metric_keys:
        metric_names.extend([f"{key}_{channel_name}" for channel_name in channel_names])
    return metric_names


if __name__ == "__main__":
    from ignite.metrics import SSIM
    import torch

    torch.manual_seed(8)
    batch_size = 64
    channels = 3
    img_size = 64
    m = MultiChannelMetric(SSIM(data_range=1.0), 3)

    y_pred = torch.rand((batch_size, channels, img_size, img_size))
    y = torch.rand((batch_size, channels, img_size, img_size))

    m.update((y_pred, y))
    res = m.compute()

    print(res)
