import wandb
import pickle
import time
from src.datahandle import *
import time
#hyper
directory = '../../extras/data/A_training_given/training_data/'
training_folder = 'training_data'
training_labels_file= 'training_norm.csv'
#
collected_directory = '../../extras/data/Z_collected_raw/' 
#
quiz_directory = '../../extras/data/C_testing_given/test_data/'
quiz_training_folder = 'test_data'
#configurations
conf_tracking = 0
seed = 0
data_shape = 'original'
parameter_init_scale = 0.01
split= 0.8
batch_size = 64
n_epochs = 5
lr = 0.0001
#dataloading object
training_object= DataLoader(
        directory,
        training_folder,
        training_labels_file
        )
collected_object = DataLoader(
        collected_directory        
        )
collected_train, collected_test = collected_object.LoadCollectedData_info(
        split=split,
        batch_size=1
        )
#train test split
train,test = training_object.LoadModelData_info(
        split = split, 
        batch_size = 1)
train = np.vstack((train,collected_train))
test= np.vstack((test[0],collected_test[0]))
loaded_X,loaded_Y = training_object.Load_batch(test)
exit()
example_batch_x,example_batch_y = training_object.Load_batch(train[0], data_shape=data_shape)
#model initialisation

#wandb tracking
if conf_tracking:
    config = {
      "model_type" : 'Same convnet as submission 2 ,leaky relu, collected, data augmentation, dropoutlayer',
      "param_initialisation_scale" : parameter_init_scale,
      "model_shape" : str(model_shape),
      "learning_rate": lr,
      "data_split": split,
      "batch_size": batch_size,
      "data_shape" : str(data_shape),
      "epochs": n_epochs,
      "Note" : "This run focuses on adding the testing data loss, and improving workflow by making predictions directly after training. LeakyRelu is used instead of Relu, and the data is augmented"
    }
    wandb.init(project="Autonomous-driving", entity="mo379",config=config)
# 
##################GPU goes buuurrrrrr#######################
if __name__ == "__main__":
    print("Begin training")






