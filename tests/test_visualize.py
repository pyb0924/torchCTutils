import matplotlib.pyplot as plt

from torchCTutils.io import clearml_scalars_parser


def test_clearml_scalars_parser(data_dir, output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    clearml_scalars_parser(data_dir / 'validation_metrics.json', ax1)
    clearml_scalars_parser(data_dir / 'training_batch_loss.json', ax2)
    fig.savefig(output_dir / 'clearml_scalars.png')
