import wandb
import pickle
import time
import jax
from jax.example_libraries import stax, optimizers
import functools
from src.datahandle import *
from src.model_cnn import *
import time
#hyper
directory = '../../extras/data/A_training_given/training_data/'
training_folder = 'training_data'
training_labels_file= 'training_norm.csv'
#configurations
conf_tracking = 1
seed = 0
rng = jax.random.PRNGKey(seed)
data_shape = 'original'
parameter_init_scale = 0.01
split= 0.8
batch_size = 256
n_epochs = 30
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
example_batch_x,example_batch_y = training_object.Load_batch(train[0], data_shape=data_shape)
#model initialisation
print('-> Model init')
init_fun, apply_fun = my_combinator(
    stax.Conv(16,(10,10), padding='SAME'),Relu_layer,
    stax.MaxPool((3,3)),
    stax.Conv(16,(10,10), padding='SAME'),Relu_layer,
    stax.AvgPool((3,3)),
    stax.Conv(16, (5,5),padding='SAME'),Relu_layer,
    stax.MaxPool((5,5)),

    my_Flatten(),
    my_Dense(2)
)
input_shape =example_batch_x.shape[-3:]
_, params= init_fun(rng, (batch_size,) + input_shape)
model_shape = jax.tree_map(lambda x: x.shape, params)


#wandb tracking
if conf_tracking:
    config = {
      "model_type" : 'Bigger convnet + avg pool and max pool layers',
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
@jax.jit
def update(opt_state, x,y):
    loss, grads = jax.value_and_grad(loss_fn)(opt_get_params(opt_state), 
            x,y)
    opt_state =opt_update(i, grads, opt_state)
    return loss,opt_state
#optimizer_init
print('-> Optimizer init')
opt_init, opt_update, opt_get_params = optimizers.adam(lr)
opt_state = opt_init(params)
##################GPU goes buuurrrrrr#######################
if __name__ == "__main__":
    print("Begin training")
    start = time.time()
    for epoch in range(n_epochs):
        for i in range(len(train)): 
            X_batch,Y_batch=training_object.Load_batch(train[i], data_shape=data_shape) 
            loss, opt_state = update(opt_state, X_batch, Y_batch)
            print(f"- Batch {i} at loss: {loss}")
            if conf_tracking==1:
                wandb.log({"batch_loss": loss})
        print(f"--- Epoch {epoch} at loss {loss}")
    end = time.time()
    print(f"total time: {end-start}")
    pickle.dump(opt_get_params(opt_state), open('pkls/final_params.pkl', 'wb'))





