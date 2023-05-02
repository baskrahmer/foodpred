from optimum.intel import INCQuantizer


def quantize(cfg, trainer, train_loader, val_loader):
    model = trainer.model
    model.config = model.base_model.config
    accuracy_criterion = AccuracyCriterion(tolerable_loss=0.01)
    tuning_criterion = TuningCriterion(max_trials=600)
    conf = PostTrainingQuantConfig(
        approach="static", backend="default", tuning_criterion=tuning_criterion,
        accuracy_criterion=accuracy_criterion
    )

    q_model = fit(model=model, conf=conf, calib_dataloader=val_loader, eval_func=eval_func)
    model = trainer.model
    model.config = model.base_model.config

    # Load the quantization configuration detailing the quantization we wish to apply
    quantization_config = PostTrainingQuantConfig(approach="static")

    # Generate the calibration dataset needed for the calibration step
    quantizer = INCQuantizer.from_pretrained(model)

    # Apply static quantization and save the resulting model
    quantizer.quantize(
        quantization_config=quantization_config,
        calibration_dataset=train_loader,
        save_directory=cfg.save_dir,
    )

    return model
