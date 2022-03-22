def init_tracker(conf_tracking):
    if conf_tracking:
        config = {
          "model_type" : 'Basic MLP',
          "param_initialisation_scale" : parameter_init_scale,
          "model_shape" : str(model_shape),
          "learning_rate": lr,
          "data_split": split,
          "batch_size": batch_size,
          "data_shape" : str(data_shape),
          "epochs": n_epochs,
        }
        wandb.init(project="Autonomous-driving", entity="mo379",config=config)
def loss_logger(conf_tracking,loss):
    if conf_tracking==1:
        wandb.log({"batch_loss": loss})
