library(png)
img <- readPNG("leopard_small.png")
image(img[,,1])

input_image <- img[,,1]
input_test <- array(runif(25), dim = c(5, 5))
filter_edge <- matrix(c(1,0,1, 0,0,0, -1,0,1), 3, 3)
filter_edge_extreme <- matrix(c(-1,-1,-1, -1,8,-1, -1,-1,-1), 3, 3)
filters_test <- list(filter_edge, filter_edge_extreme)

conv_with_filter <- function (input, filters){
  for(filter in filters){

    ysteps <- (nrow(input) - nrow(filter)) + 1
    xsteps <- (ncol(input) - ncol(filter)) + 1

    conv_matrix <- matrix(0, nrow = xsteps, ncol = ysteps)

    for(ypos in 1:ysteps){
      for(xpos in 1:xsteps){
        input_submatrix <- input[xpos:(xpos + ncol(filter) - 1), ypos:(ypos + nrow(filter) - 1)]
        conv_matrix[xpos, ypos] <-  sum(input_submatrix * filter)
      }
    }
    print(conv_matrix)
    image(conv_matrix)
  }
}

conv_with_filter(input_image, filters_test)