# from pathlib import Path
# from clearml import Task

# from ignite.engine import Events
# from ignite.contrib.handlers.clearml_logger import (
#     ClearMLLogger,
#     global_step_from_engine,
# )


# def setup_clearML_config(config):
#     task = Task.init(
#         project_name=config.project_name,
#         task_name=config.task_name,
#     )
#     task.connect_configuration(config)
#     if config.hyper_params is not None:
#         task.connect({k: config[k] for k in config.hyper_params})
#     return task


# def get_dirname_from_config(config) -> Path:
#     dirname = Path(config.output_path) / config.project_name / config.task_name
#     dirname.mkdir(parents=True, exist_ok=True)
#     return dirname


# def setup_clearMLLogger(
#     project_name,
#     task_name,
#     trainer,
#     train_evaluator,
#     validation_evaluator,
#     optimizer,
#     metric_names,
#     log_every=100,
# ):
#     clearml_logger = ClearMLLogger(project_name=project_name, task_name=task_name)

#     # Attach the logger to the trainer to log training loss
#     clearml_logger.attach_output_handler(
#         trainer,
#         event_name=Events.ITERATION_COMPLETED(every=log_every),
#         tag="training",
#         output_transform=lambda loss: {"batchloss": loss},
#     )

#     # Attach the logger to log loss and accuracy for both training and validation
#     for tag, evaluator in [
#         ("training metrics", train_evaluator),
#         ("validation metrics", validation_evaluator),
#     ]:
#         clearml_logger.attach_output_handler(
#             evaluator,
#             event_name=Events.EPOCH_COMPLETED,
#             tag=tag,
#             metric_names=metric_names,
#             global_step_transform=global_step_from_engine(trainer),
#         )

#     # Attach the logger to the trainer to log optimizer's parameters, e.g. learning rate
#     clearml_logger.attach_opt_params_handler(
#         trainer,
#         event_name=Events.ITERATION_COMPLETED(every=log_every),
#         optimizer=optimizer,
#     )
#     return clearml_logger
