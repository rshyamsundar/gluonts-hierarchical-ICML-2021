# This script defines the "bights" object, i.e. the big hierarchical time series, with associated functions.
make.data <- function(obj_bights, list_subsets, H){
  list_basef <- lapply(list_subsets, function(ts_split){
    subseries(obj_bights, ts_split = ts_split, H = H, do.forecast = T, keep.data = F)
  })
  browser()
  Yhat <- simplify2array(lapply(list_basef, "[[", "predictions"))
  Y <- simplify2array(lapply(list_basef, "[[", "future_data"))
  list(Yhat = Yhat, Y = Y)
}

bights <- function(bts, A) {
  nbts <- ncol(bts)
  naggts <- nrow(A)
  nts <- naggts + nbts
  Tobs <- nrow(bts)
  
  A <- methods::as(A, "sparseMatrix")
  S <- rbind(A, Diagonal(nbts))
  S <- methods::as(S, "sparseMatrix")
  
  yts <- matrix(NA, nrow = nrow(bts), ncol = nts)
    
  if (nbts <= 1L) {
    stop("Argument bts must be a multivariate time series.", call. = FALSE)
  }
  
  yts[, seq(naggts)] <-  as.matrix(t(A %*% t(bts)))
  yts[, seq(naggts + 1, nts)] <- bts
  
  output <- structure(
    list(yts = yts, A = A, S = S, nbts = nbts, naggts = naggts, nts = nts, Tobs = Tobs),
    class = c("bights")
  )
  return(output)
}


subseries <- function(bights, ts_split, H = 1, do.forecast = FALSE, keep.data = FALSE) {
  
  learn_data <- bights$yts[seq(ts_split[1], ts_split[2]), ]
  
  interval_future <- ts_split[2] + seq(1, H)
  
  future_data <- bights$yts[interval_future, , drop = F]
  
  predictions <- NULL
  if(do.forecast){
    predictions <- sapply(seq(bights$nts), function(j){
      model <- fit_fct(learn_data[, j], "ARIMA")
      forecast(model, h = H)$mean
    })
  }
  
  output <- list(future_data = future_data, predictions = predictions)
  
  
  if(keep.data)
  output <- c(output, list(learn_data = learn_data))
  
  return(output)
}
