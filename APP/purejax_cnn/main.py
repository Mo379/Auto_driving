import wandb
import pickle
import time
import jax
from jax.example_libraries import stax, optimizers
import functools
from src.datahandle import *
from src.model_cnn import *
#hyper
directory = '/Users/M/Desktop/Uni_MLIS_car/extras/data/A_training_given/training_data/'
training_folder = 'training_data'
training_labels_file= 'training_norm.csv'
#configurations
conf_tracking = 0
seed = 0
rng = jax.random.PRNGKey(seed)
data_shape = 'original'
parameter_init_scale = 0.01
split= 0.8
batch_size = 256
n_epochs = 25
lr = 0.0001


#dataloading object
training_object= DataLoader(
        directory,
        training_folder,
        training_labels_file
        )
#train test split
train,test = training_object.LoadModelData_info(
        split = split, 
        batch_size = batch_size)
#loaded data iterator
loaded_X_batches, loaded_Y_batches= training_object.Load_all_batches(
        train[0:5], 
        data_shape=data_shape
        )
#model initialisation
init_fun, apply_fun = my_combinator(
    stax.Conv(32,(5,5), padding='SAME'),Relu_layer,
    stax.Conv(32,(5,5), padding='SAME'),Relu_layer,
    stax.Conv(32, (5,5),padding='SAME'),Relu_layer,
    my_Flatten(),
    my_Dense(2)
)
input_shape =loaded_X_batches.shape[-3:]
_, params= init_fun(rng, (batch_size,) + input_shape)
model_shape = jax.tree_map(lambda x: x.shape, params)



#wandb tracking
if conf_tracking:
    config = {
      "model_type" : 'Basic convnet',
      "param_initialisation_scale" : parameter_init_scale,
      "model_shape" : str(model_shape),
      "learning_rate": lr,
      "data_split": split,
      "batch_size": batch_size,
      "data_shape" : str(data_shape),
      "epochs": n_epochs,
    }
    wandb.init(project="Autonomous-driving", entity="mo379",config=config)
# 
def loss_fn(params, x, y):
    predictions = apply_fun(params,x)
    return (1/len(x))*jnp.sum((predictions-y)**2)
def TrainModelInBatches(X,Y,epochs,opt_state):
  for epoch in range(epochs):
    X_iter = iter(X)
    Y_iter = iter(Y)
    for _ in range(len(X)): 
        X_batch,Y_batch= next(X_iter),next(Y_iter)
        loss, grads = jax.value_and_grad(loss_fn)(opt_get_params(opt_state), 
                X_batch,Y_batch)
        opt_state = 1 #opt_update(i, grads, opt_state)
        print(loss)
        continue
        if conf_tracking==1:
            wandb.log({"batch_loss": loss})
  return opt_state
#optimizer_init
opt_init, opt_update, opt_get_params = optimizers.adagrad(lr)
opt_state = opt_init(params)
loss_fn= jax.jit(loss_fn)
opt_update = jax.jit(opt_update)
opt_get_params = jax.jit(opt_get_params)
##################GPU goes buuurrrrrr#######################
if __name__ == "__main__":
    print("Begin training")
    final_state = TrainModelInBatches(
            loaded_X_batches, loaded_Y_batches,
            n_epochs,opt_state)
    print(final_state)


