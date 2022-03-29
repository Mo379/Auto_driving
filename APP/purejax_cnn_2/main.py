import wandb
import pickle
import time
import jax
from jax.example_libraries import stax, optimizers
import functools
from src.datahandle import *
from src.model_cnn import *
import time
import pandas as pd
import random
#hyper
directory = '../../extras/data/A_training_given/training_data/'
training_folder = 'training_data'
training_labels_file= 'training_norm.csv'
#
quiz_directory = '../../extras/data/C_testing_given/test_data/'
quiz_training_folder = 'test_data'
#configurations
conf_tracking = 0
seed = 0
rng = jax.random.PRNGKey(seed)
rng2 = jax.random.PRNGKey(seed)
data_shape = 'original'
parameter_init_scale = 0.01
split= 0.8
batch_size = 256
n_epochs = 1
lr = 0.001
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
test = test[:,:-7,:].reshape(10,-1,4)
print('-> Model init')
def make_net(mode: str):
    return stax.serial( 
        stax.Conv(5,(5,5), padding='SAME'),LeakyRelu_layer,stax.Dropout(0.2, mode=mode),
        stax.MaxPool((2,2)),

        stax.Conv(5,(5,5), padding='SAME'),LeakyRelu_layer,stax.Dropout(0.2, mode=mode),
        stax.AvgPool((2,2)),
        
        stax.Conv(5, (5,5),padding='SAME'),LeakyRelu_layer,stax.Dropout(0.2, mode=mode),
        stax.MaxPool((2,2)),

        stax.Conv(5, (5,5),padding='SAME'),LeakyRelu_layer,stax.Dropout(0.2, mode=mode),
        stax.MaxPool((2,2)),

        stax.Conv(5, (5,5),padding='SAME'),LeakyRelu_layer,stax.Dropout(0.2, mode=mode),
        stax.MaxPool((2,2)),

        my_Flatten(),
        my_Dense(2)
    )
init_fun, apply_fun = make_net('train')
input_shape =example_batch_x.shape[-3:]
_, params= init_fun(rng, (batch_size,) + input_shape)
model_shape = jax.tree_map(lambda x: x.shape, params)


#wandb tracking
if conf_tracking:
    config = {
      "model_type" : 'Bigger convnet + avg pool and max pool layers, batch normalisation and dropout, using my own data generated from the car and data augmentation, Realu swapped for leaky relu',
      "param_initialisation_scale" : parameter_init_scale,
      "model_shape" : str(model_shape),
      "learning_rate": lr,
      "data_split": split,
      "batch_size": batch_size,
      "data_shape" : str(data_shape),
      "epochs": n_epochs,
      "Note" : "This run focuses on adding the testing data loss, and improving workflow by making predictions directly after training. The cnn sees improvments by adding additional layers such as batch normalisation and dropout. Relu swapped with LeakyRelu"
    }
    wandb.init(project="Autonomous-driving", entity="mo379",config=config)
# 
@jax.jit
def loss_fn(params, x, y):
    predictions = apply_fun(params,x,rng=rng2)
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
            #X_batch,Y_batch = training_object.Augment_batch(X_batch,Y_batch)
            loss, opt_state = update(opt_state, X_batch, Y_batch)
            print(f"- Batch {i} at loss: {loss}")
            if conf_tracking==1:
                test_x, test_y=training_object.Load_batch(test[random.randint(0, 9)], 
                        data_shape=data_shape)
                test_loss = loss_fn(opt_get_params(opt_state), test_x,test_y)
                wandb.log({"test_loss": test_loss})
                wandb.log({"batch_loss": loss})
        print(f"--- Epoch {epoch} at loss {loss}")
    end = time.time()
    print(f"total time: {end-start}")
    pickle.dump(opt_get_params(opt_state), open('pkls/final_params.pkl', 'wb'))
    print("-> Making predictions")
    init_fun, apply_fun = make_net('test')

    quiz_object= DataLoader(
        quiz_directory,
        quiz_training_folder,
        )
    quiz_train = quiz_object.LoadQuizData_info()
    X,image_order = Load_batch_quiz(quiz_train, data_shape=data_shape)
    prds = np.array(apply_fun(params,X,rng=rng2))
    final_prd = np.column_stack((image_order,prds))
    final_ordered = final_prd[final_prd[:, 0].argsort()]
    df = pd.DataFrame(final_ordered, columns = ['image_id','angle','speed'])
    df = df.astype({'image_id': 'int32'})
    df.to_csv('submission.csv', index=False,)






