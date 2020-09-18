#Load libs
library(imager)
library(abind)

img_gray <- grayscale(boats)[,,,1]
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

filters_matrix <- array(c(filter_edge, filter_edge_extreme), dim = c(3,3,2))
dim(abind(filter_edge, filter_edge_extreme,rev.along=0))

conv_with_filter <- function (inputs, filters){

  input_amount <- tail(dim(inputs), n=1)
  filter_amount <- tail(dim(filters), n=1)
  ysteps <- (nrow(inputs) - nrow(filters)) + 1
  xsteps <- (ncol(inputs) - ncol(filters)) + 1
  result_matrix <- array(0, dim = c(ysteps, xsteps, filter_amount))

  for(input_iter in 1:input_amount){

    for(filter_iter in 1:filter_amount){
      filter <- filters[,,filter_iter]
      conv_matrix <- matrix(0, nrow = ysteps, ncol = xsteps)

      for(ypos in 1:ysteps){
        for(xpos in 1:xsteps){
          input_submatrix <- inputs[ypos:(ypos + nrow(filter) - 1), xpos:(xpos + ncol(filter) - 1)]
          conv_matrix[ypos, xpos] <-  sum(input_submatrix * filter)
        }
      }

      result_matrix[,,filter_iter] <- conv_matrix
    }
  }

  return(result_matrix)
}

result <- conv_with_filter(input_image, filters_matrix)
image(result[,,1])
