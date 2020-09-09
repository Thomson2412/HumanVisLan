#----------------------install------------------------#
#Install packages
#install.packages("keras")
#install.packages("kerasR")
#--------------------install_end----------------------#


#----------------------init------------------------#

#Load libs
library(keras)
library(kerasR)

#Make sure kerasR can find python??? IDK
#kerasR::keras_init()

#Make sure keras is installed
#install_keras()

#Load mnist dataset into var
mnist <- dataset_mnist()

#Copy train into own var keep the original
x_train_base <- mnist$train$x
y_train_base <- mnist$train$y

#Copy test into own var keep the original
x_test_base <- mnist$test$x
y_test_base <- mnist$test$y

#--------------------init_end----------------------#


#----------------------Data_prep------------------------#

#Flatten x train and test from 28*28 to 784
x_train_flatten <- array_reshape(x_train_base, c(nrow(x_train_base), 784))
x_test_flatten <- array_reshape(x_test_base, c(nrow(x_test_base), 784))

#Rescale between 0 and 1 (value/255, as 255 is the max value)
#Round to create values only of 0 or 1
x_train_flatten_rounded <- round(x_train_flatten / 255, 0)
x_test_flatten_rounded <- round(x_test_flatten / 255, 0)

#Convert labels from class vector to binary matrix
y_train_categorical <- keras::to_categorical(y_train_base)
y_test_categorical <- keras::to_categorical(y_test_base)

#----------------------Data_prep_end---------------------#


#---------------------Model_default-----------------------#

#Create MLP model
model_default <- keras_model_sequential()
model_default <- layer_dense(object = model_default, units = 256, input_shape = 784)
model_default <- layer_dense(object = model_default, units = 10, activation = 'softmax')

#Compile model
model_default <- compile(
  object = model_default,
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = 'accuracy'
)

#Fitting the model
history_model_default <- fit(
  object = model_default,
  x = x_train_flatten_rounded,
  y = y_train_categorical,
  batch_size = 128,
  epochs = 12,
  verbose = 1,
  validation_split = 0.2,
  callback_tensorboard(log_dir = "logs/model_default")
)
print(history_model_default)

#Generate model score
score_model_default <- evaluate(
  object = model_default,
  x = x_test_flatten_rounded, 
  y = y_test_categorical,
  verbose = 0
)
print(score_model_default)

#-------------------Model_default_end---------------------#


#----------------------Model_relu------------------------#

#Create MLP model with relu activation
model_relu <- keras_model_sequential()
model_relu <- layer_dense(object = model_relu, units = 256, input_shape = 784, activation = 'relu')
model_relu <- layer_dense(object = model_relu, units = 10, activation = 'softmax')

#Compile model
model_relu <- compile(
  object = model_relu,
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = 'accuracy'
)

#Fitting the model
history_model_relu <- fit(
  object = model_relu,
  x = x_train_flatten_rounded,
  y = y_train_categorical,
  batch_size = 128,
  epochs = 12,
  verbose = 1,
  validation_split = 0.2,
  callback_tensorboard(log_dir = "logs/model_relu")
)
print(history_model_relu)

#Generate model score
score_model_relu <- evaluate(
  object = model_relu,
  x = x_test_flatten_rounded, 
  y = y_test_categorical,
  verbose = 0
)
print(score_model_relu)

#--------------------Model_relu_end----------------------#


#--------------------Evaluation----------------------#

print("Evaluation default:")
summary(model_default)
print(history_model_default)
print(score_model_default)

print("Evaluation relu:")
summary(model_relu)
print(history_model_relu)
print(score_model_relu)

tensorboard("logs")

#------------------Evaluation_end--------------------#


#Old code saved because throwing away is not environmentally friendly

#x_train_reshape <- array(1:nrow(x_train_base), dim=c(1, 784))
#for(i in 1:nrow(x_train_base)){
# <- array_reshape(x_train_base[i,1:28,1:28], c(784, 1))
#}