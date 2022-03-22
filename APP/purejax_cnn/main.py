import wandb
import pickle
import time
import jax
from src.datahandle import *
from src.model_cnn import *
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
##################GPU goes buuurrrrrr#######################
if __name__ == "__main__":



    #if conf_tracking==1:
     #   wandb.log({"batch_loss": loss})


