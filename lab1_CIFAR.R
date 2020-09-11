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

#Load cifar dataset into var
cifar10 <- dataset_cifar10()

#Copy train into own var keep the original
x_train_base <- cifar10$train$x
y_train_base <- cifar10$train$y

#Copy test into own var keep the original
x_test_base <- cifar10$test$x
y_test_base <- cifar10$test$y

#--------------------init_end----------------------#


#----------------------Data_prep------------------------#

#Rescale between 0 and 1 (value/255, as 255 is the max value)
#Round to create values only of 0 or 1
x_train_rescale <- x_train_base / 255
x_test_rescale <- x_test_base / 255

#Convert labels from class vector to binary matrix
y_train_categorical <- keras::to_categorical(y_train_base)
y_test_categorical <- keras::to_categorical(y_test_base)

#----------------------Data_prep_end---------------------#



#----------------------Model_cifar------------------------#

model_dcn_cifar <- keras_model_sequential()
model_dcn_cifar <- layer_conv_2d(model_dcn_cifar, filters = 32, kernel_size = c(3,3), activation = 'relu', input_shape = c(32, 32, 3), padding = "same")
model_dcn_cifar <- layer_conv_2d(model_dcn_cifar, filters = 32, kernel_size = c(3,3), activation = 'relu')
model_dcn_cifar <- layer_max_pooling_2d(model_dcn_cifar, pool_size = c(2,2))
model_dcn_cifar <- layer_dropout(model_dcn_cifar, rate = 0.25)

model_dcn_cifar <- layer_conv_2d(model_dcn_cifar, filters = 32, kernel_size = c(3,3), activation = 'relu', padding = "same")
model_dcn_cifar <- layer_conv_2d(model_dcn_cifar, filters = 32, kernel_size = c(3,3), activation = 'relu')
model_dcn_cifar <- layer_max_pooling_2d(model_dcn_cifar, pool_size = c(2,2))
model_dcn_cifar <- layer_dropout(model_dcn_cifar, rate = 0.25)

model_dcn_cifar <- layer_flatten(model_dcn_cifar)
model_dcn_cifar <- layer_dense(model_dcn_cifar, units = 512, activation = 'relu')
model_dcn_cifar <- layer_dropout(model_dcn_cifar, rate = 0.5)
model_dcn_cifar <- layer_dense(model_dcn_cifar, units = 10, activation = 'softmax')

#Compile model
model_dcn_cifar <- compile(
  object = model_dcn_cifar,
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(lr = 0.0001, decay = 1e-6),
  metrics = 'accuracy'
)

#Fitting the model
history_model_dcn_cifar <- fit(
  object = model_dcn_cifar,
  x = x_train_rescale,
  y = y_train_categorical,
  batch_size = 32,
  epochs = 20,
  verbose = 1,
  validation_data = list(x_test_rescale, y_test_categorical),
  shuffle = TRUE,
  callback_tensorboard(log_dir = "logs/model_dcn_cifar")
)
print(history_model_dcn_cifar)

#Generate model score
score_model_dcn_cifar <- evaluate(
  object = model_dcn_cifar,
  validation_data = list(x_test_rescale, y_test_categorical),
  verbose = 0
)
print(score_model_dcn_cifar)

#--------------------Model_cifar_end----------------------#


#--------------------Evaluation----------------------#

print("Evaluation cifar:")
summary(model_dcn_cifar)
print(history_model_dcn_cifar)
print(score_model_dcn_cifar)

tensorboard("logs")

#------------------Evaluation_end--------------------#


