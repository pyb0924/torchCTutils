# import warnings
# from ignite.engine import Events
# 
# from ..io import save_multichannel_grayscale_image
# from .multimetric import get_multichannel_metric_names
# 
# 
# def print_logs_handler(trainer, pbar, timer, evaluator):
#     @trainer.on(Events.EPOCH_COMPLETED)
#     def print_times(engine):
#         pbar.log_message(
#             f"Epoch {engine.state.epoch} done. Time per batch: {timer.value():.3f}[s]")
#         timer.reset()
# 
#     @trainer.on(Events.EPOCH_COMPLETED)
#     def print_train_results():
#         metric_results = evaluator.state.metrics
#         message = 'train result:\n'
#         for metric_key, metric_value in metric_results.items():
#             message += f'\t{metric_key}:{round(metric_value, 4)}\n'
#         pbar.log_message(message)
# 
#     return print_times, print_train_results
# 
# 
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
