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

#Reshape x train and test from 28*28 to 28*28*1
x_train_3d <- array_reshape(x_train_base, c(nrow(x_train_base), 28, 28, 1))
x_test_3d <- array_reshape(x_test_base, c(nrow(x_test_base), 28, 28, 1))

#Rescale between 0 and 1 (value/255, as 255 is the max value)
#Round to create values only of 0 or 1
x_train_3d_rounded <- round(x_train_3d / 255, 0)
x_test_3d_rounded <- round(x_test_3d / 255, 0)

#Convert labels from class vector to binary matrix
y_train_categorical <- keras::to_categorical(y_train_base)
y_test_categorical <- keras::to_categorical(y_test_base)

#----------------------Data_prep_end---------------------#



#----------------------Model_deep------------------------#

model_dcn <- keras_model_sequential()
model_dcn <- layer_conv_2d(model_dcn, filters = 32, kernel_size = c(3,3), activation = 'relu', input_shape = c(28,28,1))
model_dcn <- layer_conv_2d(model_dcn, filters = 64, kernel_size = c(3,3), activation = 'relu')
model_dcn <- layer_max_pooling_2d(model_dcn, pool_size = c(2,2))
model_dcn <- layer_flatten(model_dcn)
model_dcn <- layer_dense(model_dcn, units = 128, activation = 'relu')
model_dcn <- layer_dense(model_dcn, units = 10, activation = 'softmax')

#Compile model
model_dcn <- compile(
  object = model_dcn,
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adadelta(),
  metrics = 'accuracy'
)

#Fitting the model
history_model_dcn <- fit(
  object = model_dcn,
  x = x_train_3d_rounded,
  y = y_train_categorical,
  batch_size = 128,
  epochs = 6,
  verbose = 1,
  validation_split = 0.2,
  callback_tensorboard(log_dir = "logs/model_dcn")
)
print(history_model_dcn)

#Generate model score
score_model_dcn <- evaluate(
  object = model_dcn,
  x = x_test_3d_rounded, 
  y = y_test_categorical,
  verbose = 0
)
print(score_model_dcn)


#--------------------Model_deep_end----------------------#


#----------------------Model_drop------------------------#

model_dcn_dropout <- keras_model_sequential()
model_dcn_dropout <- layer_conv_2d(model_dcn_dropout, filters = 32, kernel_size = c(3,3), activation = 'relu', input_shape = c(28,28,1))
model_dcn_dropout <- layer_conv_2d(model_dcn_dropout, filters = 64, kernel_size = c(3,3), activation = 'relu')
model_dcn_dropout <- layer_max_pooling_2d(model_dcn_dropout, pool_size = c(2,2))
model_dcn_dropout <- layer_dropout(model_dcn_dropout, rate = 0.25)
model_dcn_dropout <- layer_flatten(model_dcn_dropout)
model_dcn_dropout <- layer_dense(model_dcn_dropout, units = 128, activation = 'relu')
model_dcn_dropout <- layer_dropout(model_dcn_dropout, rate = 0.5)
model_dcn_dropout <- layer_dense(model_dcn_dropout, units = 10, activation = 'softmax')

#Compile model
model_dcn_dropout <- compile(
  object = model_dcn_dropout,
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adadelta(),
  metrics = 'accuracy'
)

#Fitting the model
history_model_dcn_dropout <- fit(
  object = model_dcn_dropout,
  x = x_train_3d_rounded,
  y = y_train_categorical,
  batch_size = 128,
  epochs = 6,
  verbose = 1,
  validation_split = 0.2,
  callback_tensorboard(log_dir = "logs/model_dcn_dropout")
)
print(history_model_dcn_dropout)

#Generate model score
score_model_dcn_dropout <- evaluate(
  object = model_dcn_dropout,
  x = x_test_3d_rounded, 
  y = y_test_categorical,
  verbose = 0
)
print(score_model_dcn_dropout)

#--------------------Model_drop_end----------------------#


#--------------------Evaluation----------------------#

print("Evaluation dcn:")
summary(model_dcn)
print(history_model_dcn)
print(score_model_dcn)

print("Evaluation dropout:")
summary(model_dcn_dropout)
print(history_model_dcn_dropout)
print(score_model_dcn_dropout)

tensorboard("logs")

#------------------Evaluation_end--------------------#


#Old code saved because throwing away is not environmentally friendly

#x_train_reshape <- array(1:nrow(x_train_base), dim=c(1, 784))
#for(i in 1:nrow(x_train_base)){
# <- array_reshape(x_train_base[i,1:28,1:28], c(784, 1))
#}