from typing import Any, Optional

from ignite.engine import Engine
from ignite.engine.events import Events
from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine
from ignite.handlers.early_stopping import EarlyStopping


def setup_handlers(
    trainer: Engine,
    evaluator: Engine,
    config: Any,
    to_save_train: Optional[dict] = None,
    to_save_eval: Optional[dict] = None,
):
    """Setup Ignite handlers."""

    ckpt_handler_train = ckpt_handler_eval = None
    # checkpointing
    saver = DiskSaver(config.output_dir / "checkpoints", require_empty=False)
    ckpt_handler_train = Checkpoint(
        to_save_train,
        saver,
        filename_prefix=config.filename_prefix,
        n_saved=config.n_saved,
    )
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED(every=config.save_every_iters),
        ckpt_handler_train,
    )
    global_step_transform = None
    if to_save_train.get("trainer", None) is not None:
        global_step_transform = global_step_from_engine(to_save_train["trainer"])
    ckpt_handler_eval = Checkpoint(
        to_save_eval,
        saver,
        filename_prefix="best",
        n_saved=config.n_saved,
        global_step_transform=global_step_transform,
        score_name="model_d_error",
        score_function=Checkpoint.get_default_score_fn("errD", -1),
    )
    evaluator.add_event_handler(Events.EPOCH_COMPLETED(every=1), ckpt_handler_eval)

    # early stopping
    def score_fn(engine: Engine):
        return -engine.state.metrics["errD"]

    es = EarlyStopping(config.patience, score_fn, trainer)
    evaluator.add_event_handler(Events.EPOCH_COMPLETED, es)
    return ckpt_handler_train, ckpt_handler_eval


# def save_example_factory(trainer, model, output_dir, checkpoint_every, device):
#     @trainer.on(Events.EPOCH_COMPLETED(every=checkpoint_every))
#     def save_example(engine):
#         image, label = engine.state.batch
#         output = model(image.to(device))
#
#         output_filenames = [
#             output_dir / 'images' / material / f'output_{engine.state.epoch}.png' for material in GT_TYPES]
#         save_multichannel_grayscale_image(
#             output, output_filenames, normalize=True)
#
#         label_filenames = [
#             output_dir / 'images' / material / f'label_{engine.state.epoch}.png' for material in GT_TYPES]
#         save_multichannel_grayscale_image(
#             label, label_filenames, normalize=True)
#
#     return save_example
#
#
# def run_evaluator_handler(trainer, checkpoint_every, output_dir, evaluator, valid_loader, pbar):
#     @trainer.on(Events.EPOCH_COMPLETED(every=checkpoint_every))
#     def run_validation(engine):
#         evaluator.run(valid_loader)
#
#         metric_results = evaluator.state.metrics
#         message = 'Validation result:\n'
#         for metric_key, metric_value in metric_results.items():
#             message += f'\t{metric_key}:{list(map(lambda x: round(x, 4), metric_value))}\n'
#         pbar.log_message(message)
#
#         fname = output_dir / 'validations.csv'
#         columns = ["epoch"] + get_multichannel_metric_names(metric_results.keys(), GT_TYPES)
#         values = [str(engine.state.epoch)]
#         for value in metric_results.values():
#             values += list(map(lambda x: str(round(x, 5)), value))
#
#         with open(fname, "a") as f:
#             if f.tell() == 0:
#                 print(",".join(columns), file=f)
#             print(",".join(values), file=f)
#
#     return run_validation
#
#
# def handle_exception_handler(trainer):
#     @trainer.on(Events.EXCEPTION_RAISED)
#     def handle_exception(engine, e):
#         if isinstance(e, KeyboardInterrupt) and (engine.state.iteration > 1):
#             engine.terminate()
#             warnings.warn("KeyboardInterrupt caught. Exiting gracefully.")
#         else:
#             raise e
#
#     return handle_exception
