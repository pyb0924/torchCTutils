from ignite.contrib.handlers.wandb_logger import WandBLogger

def setup_wandbLogger(project_name, task_name, trainer, train_evaluator, validation_evaluator, model, optimizer, metric_names, log_every=100):
    wandb_logger = WandBLogger(
        project=project_name,
        name=task_name,
    )

    wandb_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED(every=log_every),
        tag="training",
        output_transform=lambda loss: {"batchloss": loss},
    )

    for tag, evaluator in [("training", train_evaluator), ("validation", validation_evaluator)]:
        wandb_logger.attach_output_handler(
            evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag=tag,
            metric_names=metric_names,
            global_step_transform=lambda *_: trainer.state.iteration,
        )

    wandb_logger.attach_opt_params_handler(
        trainer, event_name=Events.ITERATION_COMPLETED(every=log_every), optimizer=optimizer
    )
    wandb_logger.watch(model, log="all")