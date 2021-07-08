# All the functions needed to compute the MinT forecasts.
compute_pmint <- function(objhts, method = "diagonal", residuals){
  J <- Matrix(cbind(matrix(0, nrow = objhts$nbts, ncol = objhts$nts - objhts$nbts), diag(objhts$nbts)), sparse = TRUE)
  U <- Matrix(rbind(diag(objhts$nts - objhts$nbts), -t(objhts$A)), sparse = TRUE)
  P_BU <- cbind(matrix(0, objhts$nbts, objhts$nts - objhts$nbts), diag(objhts$nbts)) 
  
  #if(!is.null(residuals))
  #  R1 <- t(residuals)
  
  R1 <- t(residuals)
  
  if(is.null(method))
    method <- "diagonal"
  
  if(method == "diagonal"){
    # Diagonal matrix
    W <- Diagonal(x = vec_w(R1))
  }else if(method == "shrink"){
    # Shrunk matrix
    target_diagonal <- lowerD(R1)
    shrink_results <- shrink.estim(R1, target_diagonal)
    W <- shrink_results$shrink.cov
  }else if(method == "ols"){
    W <- diag(objhts$nts)
  }else if(method == "sample"){
    n <- nrow(R1)
    W <- crossprod(R1) / n
    if(is.positive.definite(W)==FALSE)
    {
      stop("MinT needs covariance matrix to be positive definite.", call. = FALSE)
    }
  }
  
  MAT1 <- W %*% U
  MAT2 <- crossprod(U,MAT1)
  MAT3 <- tcrossprod(solve(MAT2), U)
  C1 <- J %*% MAT1
  P_MINT <- P_BU - C1 %*% MAT3
  
  S <- objhts$S
  n_agg <- objhts$naggts
  n_total <- objhts$nts
  V <- S %*% P_MINT %*% W %*% t(P_MINT) %*% t(S)
  
  #V_agg <- diag(V)[seq(n_agg)]
  #V_bot <- diag(V)[seq(n_agg + 1, n_total)]
  #list(V_agg = V_agg, V_bot = V_bot)
  
  
  list(P_MINT = P_MINT, W = W, V = V)
}

shrink.estim <- function(x, tar)
{
  if (is.matrix(x) == TRUE && is.numeric(x) == FALSE)
    stop("The data matrix must be numeric!")
  p <- ncol(x)
  n <- nrow(x)
  
  covm <- crossprod(x) / n
  corm <- cov2cor(covm)
  xs <- scale(x, center = FALSE, scale = sqrt(diag(covm)))
  v <- (1/(n * (n - 1))) * (crossprod(xs^2) - 1/n * (crossprod(xs))^2)
  diag(v) <- 0
  corapn <- cov2cor(tar)
  d <- (corm - corapn)^2
  lambda <- sum(v)/sum(d)
  lambda <- max(min(lambda, 1), 0)
  shrink.cov <- lambda * tar + (1 - lambda) * covm
  return(list(shrink.cov = shrink.cov, lambda = lambda ))
}

vec_w <- function(x){
	n <- nrow(x)
	apply(x, 2, crossprod) / n
}
lowerD <- function(x)
{
  n <- nrow(x)
  return(diag(vec_w(x)))
}
