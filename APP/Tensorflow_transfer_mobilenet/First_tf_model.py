#!/usr/bin/env python
# coding: utf-8

# # Importing pakages

# In[1]:


import wandb
import pickle
import time
from src.datahandle import *
import time


# # Setting up hyperparams

# In[16]:


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
batch_size = 32
n_epochs = 5
lr = 0.0001


# # Setting up DataLoaders

# In[17]:


#dataloading object
training_object= DataLoader(
        directory,
        training_folder,
        training_labels_file
        )
collected_object = DataLoader(
        collected_directory        
        )


# # Loading data information (not loading images yet)

# In[18]:


collected_train, collected_test = collected_object.LoadCollectedData_info(
        split=split,
        batch_size=batch_size #for tf this batch size remains as 1
        )
#train test split
train,test = training_object.LoadModelData_info(
        split = split, 
        batch_size =batch_size)#for tf this batch size remains as 1


# # Stacking collected and kaggle given data

# In[19]:


train_XY = np.vstack((train,collected_train))
test_XY= np.vstack((test[0],collected_test[0]))


# # Checking DataLoading functionality 

# In[20]:


for i in range(1):
    X,Y = training_object.Load_batch(train_XY[i], data_shape=data_shape)
image_shape = X.shape[1:]
image_shape


# # Setting up data augmentation

# In[1]:


data_augmentation = keras.Sequential(
  [
        layers.experimental.preprocessing.RandomRotation(0.2),
        layers.experimental.preprocessing.RandomZoom(0.2),
        #layers.experimental.preprocessing.RandomContrast(0.2),
  ]
)


# # Initialising the model

# In[ ]:


#model initialisation
model = keras.Sequential(
    [
        keras.Input(shape=image_shape),
        data_augmentation,
        layers.Conv2D(32, 5, strides=2, activation="relu"),
        data_augmentation,
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(2)
    ]
)
model.compile(optimizer='adam',
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['accuracy'])


# In[9]:


model.layers


# # Weights and Biases configuration

# In[15]:


#wandb tracking
if conf_tracking:
   config = {
     "model_type" : 'Same convnet as submission 2 ,leaky relu, collected, data augmentation, dropoutlayer',
     "param_initialisation_scale" : parameter_init_scale,
     "model_shape" : str(model.layers),
     "learning_rate": lr,
     "data_split": split,
     "batch_size": batch_size,
     "data_shape" : str(data_shape),
     "epochs": n_epochs,
   }
   run = wandb.init(project="Autonomous-driving", entity="mo379",config=config)


# # Training Loop (With wandb logging)

# In[11]:


model_path = 'pkls/tf_cnn_augmented.pkl'
#model = tf.keras.models.load_model(model_path, custom_objects=None, compile=True, options=None)
test_batches = test_XY[0:2930].reshape(10,-1,4)
for _ in range(n_epochs):
    for i in range(len(train_XY)):
        X,Y = training_object.Load_batch(train_XY[i], data_shape=data_shape)
        train_metrics = model.train_on_batch(X,Y, return_dict=True) 

        X,Y = training_object.Load_batch(test_batches[np.random.randint(0,9)], data_shape=data_shape)
        test_metrics = model.test_on_batch(
            X, Y, sample_weight=None, reset_metrics=True, return_dict=True
        )
        
        if conf_tracking==1:
            wandb.log({"test_loss": test_metrics['loss']})
            wandb.log({"batch_loss": train_metrics['loss']})

            wandb.log({"test_accuracy": test_metrics['accuracy']})
            wandb.log({"batch_accuracy": train_metrics['accuracy']})
#tf.keras.models.save_model(
#    model,
#    model_path,
#    overwrite=False,
#    include_optimizer=True,
#    save_format=None,
#    signatures=None,
#    options=None,
#    save_traces=True
#)


# # Loading the Quiz data

# In[ ]:


quiz_object= DataLoader(
    quiz_directory,
    quiz_training_folder,
)
quiz_train = quiz_object.LoadQuizData_info()
X,image_order = quiz_object.Load_batch_quiz(quiz_train,data_shape=data_shape)


# # Making predictions

# In[ ]:


prds = model.predict(X)
final_prd = np.column_stack((image_order,prds))
final_ordered = final_prd[final_prd[:, 0].argsort()]
df = pd.DataFrame(final_ordered, columns = ['image_id','angle','speed'])
df = df.astype({'image_id': 'int32'})
df.to_csv('submission.csv', index=False,)


# In[ ]:




