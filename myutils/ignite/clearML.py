from clearml import Task

from ignite.engine import Events
from ignite.contrib.handlers.clearml_logger import (
    ClearMLLogger,
    GradsHistHandler,
    GradsScalarHandler,
    WeightsHistHandler,
    WeightsScalarHandler,
    global_step_from_engine
)


def setup_clearML_config(config):
    task = Task.init(project_name=config.project_name, task_name=config.task_name, output_uri=config.output_path)
    task.connect_configuration(config)
    if config.hyper_params is not None:
        task.connect({k: config[k] for k in config.hyper_params})
    return task


def setup_clearMLLogger(project_name, task_name, trainer, train_evaluator, validation_evaluator, model, optimizer, metric_names, log_every=100):
    clearml_logger = ClearMLLogger(
        project_name=project_name, task_name=task_name
    )

    # Attach the logger to the trainer to log training loss
    clearml_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED(every=log_every),
        tag="training",
        output_transform=lambda loss: {"batchloss": loss},
    )

    # Attach the logger to log loss and accuracy for both training and validation
    for tag, evaluator in [("training metrics", train_evaluator), ("validation metrics", validation_evaluator)]:
        clearml_logger.attach_output_handler(
            evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag=tag,
            metric_names=metric_names,
            global_step_transform=global_step_from_engine(trainer),
        )

    # Attach the logger to the trainer to log optimizer's parameters, e.g. learning rate
    clearml_logger.attach_opt_params_handler(
        trainer, event_name=Events.ITERATION_COMPLETED(every=log_every), optimizer=optimizer
    )

    # Attach the logger to the trainer to log model's weights norm
    clearml_logger.attach(
        trainer, log_handler=WeightsScalarHandler(model), event_name=Events.ITERATION_COMPLETED(every=log_every)
    )

    # Attach the logger to the trainer to log model's weights as a histogram
    clearml_logger.attach(trainer, log_handler=WeightsHistHandler(
        model), event_name=Events.EPOCH_COMPLETED(every=log_every))

    # Attach the logger to the trainer to log model’s gradients as scalars
    clearml_logger.attach(
        trainer, log_handler=GradsScalarHandler(model), event_name=Events.ITERATION_COMPLETED(every=log_every)
    )

    # Attach the logger to the trainer to log model's gradients as a histogram
    clearml_logger.attach(trainer, log_handler=GradsHistHandler(
        model), event_name=Events.EPOCH_COMPLETED(every=log_every))

    return clearml_logger
