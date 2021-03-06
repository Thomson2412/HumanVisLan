#Load libs
library(dplyr)
library(imager)

img_gray <- grayscale(boats)[, , , 1]
image(img_gray)

input_image <- img_gray
#input_test <- array(runif(25), dim = c(5, 5))

filter_edge <- matrix(c(
   1, 0, 1,
   0, 0, 0,
  -1, 0, 1), 3, 3)

filter_edge_extreme <- matrix(c(
  -1, -1, -1,
  -1,  8, -1,
  -1, -1, -1), 3, 3)


filter_sharpen <- matrix(c(
   0, -1,  0,
  -1,  5, -1,
   0, -1,  0), 3, 3)

conv_with_filter <- function(input_layer, input_filters) {
  #Check correctness of input layer dimensions
  if (length(dim(input_layer)) != 2 && length(dim(input_layer)) != 3) {
    print("Input not 2D or 3D")
    return(NULL)
  }
  #Check correctness of filter dimensions
  if (length(dim(input_filters)) != 3) {
    print("Filter not 3D")
    return(NULL)
  }

  #Convert to 3D if 2D
  if (length(dim(input_layer)) == 2)
    dim(input_layer) <- c(nrow(input_layer), ncol(input_layer), 1)

  #Get the amount of input layers and filters
  input_amount <- tail(dim(input_layer), n = 1)
  filter_amount <- tail(dim(input_filters), n = 1)

  #Check if the there is a filter for every input layer
  if (input_amount != filter_amount) {
    print("Amount of input layers and filter layers does not correspond")
    return(NULL)
  }

  #Calculate how many steps the filter should "walk" over the input matrix (+1 as the first postion is also a step)
  ysteps <- (nrow(input_layer) - nrow(input_filters)) + 1
  xsteps <- (ncol(input_layer) - ncol(input_filters)) + 1
  #Generate empty result matrix to be accumilated below
  result_matrix <- array(0, dim = c(ysteps, xsteps, filter_amount))

  #Apply every filter to every corresponding input
  for (filter_iter in 1:filter_amount) {
    input <- input_layer[, , filter_iter]
    filter <- input_filters[, , filter_iter]
    #Generate empty convolution matrix per filter to be accumilated below
    conv_matrix <- matrix(0, nrow = ysteps, ncol = xsteps)
    #Walk over input with filter: per row "walk" over every colum
    for (ypos in 1:ysteps) {
      for (xpos in 1:xsteps) {
        #Grab a submatrix from the input by offset of current step + size of filter x and y
        input_submatrix <- input[ypos:(ypos + nrow(filter) - 1), xpos:(xpos + ncol(filter) - 1)]
        #Sum the element multiplication of the input submatrix and the current filter to create a convolution element
        conv_matrix[ypos, xpos] <- sum(input_submatrix * filter)
      }
    }
    #Put the generated convolution into the all contioning result matrix (3D)
    result_matrix[, , filter_iter] <- conv_matrix
  }
  return(result_matrix)
}

relu_activation <- function (input_feature_map){
  if (length(dim(input_feature_map)) != 2) {
    print("Input not 2D")
    return(NULL)
  }
  return(pmax(input_feature_map, 0))
}

max_pooling <- function(input_layer, pool_y_size, pool_x_size) {
  if (length(dim(input_layer)) != 2) {
    print("Input not 2D")
    return(NULL)
  }

  #Calculate how many steps the filter should "walk" over the input matrix
  ysteps <- ceiling(nrow(input_layer) / pool_y_size)
  xsteps <- ceiling(ncol(input_layer) / pool_x_size)

  #Generate empty pool matrix to be accumilated below
  pool_matrix <- matrix(0, nrow = ysteps, ncol = xsteps)
  #Walk over input with filter: per row "walk" over every colum
  for (ypos in 1:ysteps) {
    for (xpos in 1:xsteps) {
      #Create pool dimensions, don't go over the edge
      yoffset_min <- ((ypos * pool_y_size) - pool_y_size) + 1
      xoffset_min <- ((xpos * pool_x_size) - pool_x_size) + 1
      yoffset_max <- yoffset_min + pool_y_size - 1
      xoffset_max <- xoffset_min + pool_x_size - 1
      if(yoffset_max > nrow(input_layer))
        yoffset_max <- nrow(input_layer)
      if(xoffset_max > ncol(input_layer))
        xoffset_max <- ncol(input_layer)
      #Grab a submatrix from the input by above created dimensions
      input_submatrix <- input_layer[yoffset_min:yoffset_max, xoffset_min:xoffset_max]
      #Max the input submatrix
      pool_matrix[ypos, xpos] <- max(input_submatrix)
    }
  }

  return(pool_matrix)
}

normalize_zero <- function (input_layer){
  if (length(dim(input_layer)) != 2) {
    print("Input not 2D")
    return(NULL)
  }
  scaled <- (input_layer - mean(input_layer)) / sd(input_layer)
  return(scaled)
}

fully_connected <- function(input_layers, neurons){
  input_flatten <- input_layers
  if (length(dim(input_flatten)) > 1) {
    #flatten matrix to 1D vector
    dim(input_flatten) <- NULL
  }

  hidden_layer <- array(0, dim = neurons)
  for(neuron_it in 1:neurons){
    weight_list <- array(runif(1), dim = length(input_flatten))
    hidden_layer[[neuron_it]] <- sum((input_flatten * weight_list))
  }
  return(hidden_layer)
}

softmax_activation <- function(input_layer){
  scaled <- (input_layer - mean(input_layer)) / sd(input_layer)
  result <- exp(scaled) / sum(exp(scaled))
  return(result)
}

#Run with single input image as input layer and one filter layer
single_filter_3D <- filter_edge
dim(single_filter_3D) <- c(nrow(single_filter_3D), ncol(single_filter_3D), 1)
featuremap_fe <- conv_with_filter(input_image, single_filter_3D)[, , 1]
image(img_gray)
image(featuremap_fe)

#Run with single input image as input layer and one filter layer
single_filter_3D <- filter_edge_extreme
dim(single_filter_3D) <- c(nrow(single_filter_3D), ncol(single_filter_3D), 1)
featuremap_fee <- conv_with_filter(input_image, single_filter_3D)[, , 1]
image(img_gray)
image(featuremap_fee)

#Run with single input image as input layer and one filter layer
single_filter_3D <- filter_sharpen
dim(single_filter_3D) <- c(nrow(single_filter_3D), ncol(single_filter_3D), 1)
featuremap_fs <- conv_with_filter(input_image, single_filter_3D)[, , 1]
image(img_gray)
image(featuremap_fs)


#Run with multiple(3) feature maps as input layer and multiple filter layers(3)
input_layers <-
  array(c(featuremap_fs, featuremap_fe, featuremap_fee), dim = c(nrow(featuremap_fs), ncol(featuremap_fs), 3))
filters_matrix <-
  array(c(filter_edge, filter_edge_extreme, filter_sharpen), dim = c(nrow(filter_edge), ncol(filter_edge), 3))
result_multiple <- conv_with_filter(input_layers, filters_matrix)
image(img_gray)
image(result_multiple[, , 1])
image(result_multiple[, , 2])
image(result_multiple[, , 3])

image(relu_activation(result_multiple[, , 1]))
image(relu_activation(result_multiple[, , 2]))
image(relu_activation(result_multiple[, , 3]))

pool_result <- max_pooling(result_multiple[, , 1], 8, 8)
image(pool_result)

normalize_result <- normalize_zero(pool_result)
image(normalize_result)
mean(normalize_result)
sd(normalize_result)
hist(normalize_result)

fully_result <- fully_connected(normalize_result, 10)

softmax_prob <- softmax_activation(fully_result)
sum(softmax_prob)