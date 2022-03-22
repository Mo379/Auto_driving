import wandb
import pickle
import time
from src.datahandle import *
from src.model_mlp import *
from src.wandb_track import *
#hyper
directory = '/Users/M/Desktop/Uni_MLIS_car/extras/data/A_training_given/training_data/'
training_folder = 'training_data'
training_labels_file= 'training_norm.csv'
#configurations
conf_tracking = 0
seed = 0
model_shape = [230400,512,512,2]
data_shape = (-1)
parameter_init_scale = 0.01
split= 0.8
batch_size = 256
n_epochs = 25
lr = 0.0001

#wandb tracking
init_tracker(conf_tracking)
#########################################
if __name__ == "__main__":
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
    key = jax.random.PRNGKey(seed)
    params = init_model(model_shape, key ,scale=parameter_init_scale)
    r_t = jax.tree_map(lambda x: x-x , params)
    #training loop
    for epoch in range(n_epochs):
        #data iterator
        X_iter = iter(loaded_X_batches)
        Y_iter = iter(loaded_Y_batches)
        for _ in range(len(loaded_X_batches)): 
            X,Y= next(X_iter),next(Y_iter)
            r_t,loss,params = RMSprop_update(params, X,Y, lr=lr, r_t=r_t)
            loss_logger(conf_tracking,loss)
        if epoch % int(n_epochs//10) ==0:
            print(f"loss: {loss}")
