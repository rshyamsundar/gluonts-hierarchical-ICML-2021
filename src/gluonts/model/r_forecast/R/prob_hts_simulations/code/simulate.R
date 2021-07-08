# This script contains all the functions to simulate the AR models as well as the true distribution of the future observations.
transform_sample <- function(X, mu_theo, sd_theo){
  mu <- apply(X, 2, mean)
  sdeviation <- apply(X, 2, sd)
  t( ((t(X) - mu + mu_theo)/sdeviation) * sd_theo )
}

gen.innov <- function(n, marg, meanVec, covarianceMat){
  
  mycor <- diag(diag(covarianceMat)^(-.5)) %*% covarianceMat %*% diag(diag(covarianceMat)^(-.5)) 
  
  nvar <- ncol(covarianceMat)
  correlations <- mycor[lower.tri(mycor)]
  mycop <- normalCopula(correlations, dispstr = "un", dim = nvar)
  margins <-  rep(marg, nvar)
  if(marg == "norm"){
    parameters <- lapply(seq(nvar), function(i){
      list(mean = meanVec[i], sd = sqrt(covarianceMat[i, i]) )
    })
    #mvrnorm(n = n_simul, mus, Sigma = Sigma)
  }else if(marg == "t"){
    list_param  <- list(df = 6) #list(df = 4) # df = 10 too big
    parameters <- rep(list(list_param), nvar)
  }else if(marg == "sn"){
    list_param  <- list(alpha = -5, omega = 5)
    parameters <- rep(list(list_param), nvar)
  }else if(marg == "mst"){
    list_param  <- list(alpha = -20, omega = 3, nu = 7) 
    parameters <- rep(list(list_param), nvar)
  }else if(marg == "mixture"){
    probs <- c(0.3,0.7)
    mus <-  c(-2, 2)
    
    list_param <- list(probs = probs, mus = mus, sds = sqrt(c(1,1)) )
    parameters <- rep(list(list_param), nvar)
  }
  my.mvdc <- mvdc(mycop, margins, parameters)
  innov <- rMvdc(n, my.mvdc)
  #innov <- scale(innov, scale = F)
}

orderGen <- function(n.bot)
{
  
  order.diff <- rep(0, n.bot)
  order.ar <- sample(0:2, n.bot, replace = TRUE)
  order.ma <- sample(0:2, n.bot, replace = TRUE)
  order.d <- cbind(order.ar, order.diff, order.ma)
  
  
  ar.d <- matrix(, n.bot, 2)
  ma.d <- matrix(, n.bot, 2)
  
  for(j in 1:n.bot)
  {
    order.t <- order.d[j, ]
    
    # define AR coefficients
    ar.coeff <- c()
    if(order.t[1]==0)
    {
      ar.d[j, 1] <- NA
    }
    
    ar.coeff1 <- 0
    ar.coeff2 <- 0
    if(order.t[1]==1)
    {
      ar.coeff1 <- runif(1, 0.5, 0.7)
      ar.coeff <- ar.coeff1
      ar.d[j, 1] <- ar.coeff 
    }
    if(order.t[1]==2)
    {
      ar.coeff2 <- runif(1, 0.5, 0.7)
      lower.ar.b <- ar.coeff2 - 0.9
      upper.ar.b <- 0.9 - ar.coeff2
      ar.coeff1 <- runif(1, lower.ar.b, upper.ar.b)
      ar.coeff <- c(ar.coeff1, ar.coeff2)
      ar.d[j, 1:2] <- ar.coeff
    }
    
    # define MA coefficients
    ma.coeff <- c()
    if(order.t[3]==0)
    {
      ma.d[j, 1] <- NA
    }
    ma.coeff1 <- 0
    ma.coeff2 <- 0
    if(order.t[3]==1)
    {
      ma.coeff1 <- runif(1, 0.5, 0.7)
      ma.coeff <- ma.coeff1
      ma.d[j, 1] <- ma.coeff
    }
    if(order.t[3]==2)
    {
      ma.coeff2 <- runif(1, 0.5, 0.7)
      lower.ma.b <- -1 * (0.9 + ma.coeff2) / ((0.9+0.7)/0.5)
      upper.ma.b <- -1 * lower.ma.b
      ma.coeff1 <- runif(1, lower.ma.b, upper.ma.b)
      ma.coeff <- c(ma.coeff1, ma.coeff2)
      ma.d[j, 1:2] <- ma.coeff
    }
  }
  return(list(ar.d, order.d, ma.d))
}

dataGen <- function(n, n.bot, var.mat, npaths)
{ 
  order.gen <- ordergenCor(n.bot)
  ar.d <- order.gen[[1]]
  order.d <- order.gen[[2]]
  ma.d <- order.gen[[3]]
  data.bot <- matrix(NA, n, n.bot)
  
  if(!is.matrix(var.mat)){
    nvar <- 1
    Sigma <- matrix(var.mat)
  }else{
    nvar <- n.bot
    Sigma <- var.mat
  }
  
  if(nvar == 1){ # Common pattern
    #innov_burnin   <- rmvnorm(n_warm, mean = rep(0, nvar), sigma = Sigma)
    #innov_insample <- rmvnorm(n, mean = rep(0, nvar), sigma = Sigma)
    #innov_future   <- rmvnorm(npaths * H, mean = rep(0, nvar), sigma = Sigma)
    
    innov_burnin   <- matrix(rnorm(n_warm, mean = 0, sd = Sigma), ncol = 1)
    innov_insample <- matrix(rnorm(n, mean = 0, sd = Sigma), ncol = 1)
    #innov_future   <- matrix(rnorm(npaths * H, mean = 0, sd = Sigma), ncol = 1)
    
    innov_future <- sapply(seq(H), function(h){
      x <- matrix(rnorm(npaths, mean = 0, sd = Sigma), ncol = 1)
      transform_sample(x, rep(0, nvar), sqrt(diag(Sigma)) )
    }, simplify = "array")

  }else{
    
    allsd  <- sqrt(diag(Sigma))
    innov_burnin <- matrix(NA, nrow = n_warm, ncol = nvar)
    innov_insample <- matrix(NA, nrow = n, ncol = nvar)
    innov_future <- array(NA, c(npaths, nvar, H))
    for(j in seq(nvar)){
      innov_burnin[, j] <- rnorm(n_warm, mean = 0, sd = allsd[j])
      innov_insample[, j] <- rnorm(n, mean = 0, sd = allsd[j])
      innov_future[, j , ] <- matrix(rnorm(npaths * H, mean = 0, sd = allsd[j]), nrow = npaths, ncol = H)
    }
    
  }
  
  list_paths <- vector("list", nvar)
  for(j in 1:nvar)
  {
    
    # generating data from a ARIMA model
    data.bot[, j] <- arima.sim(list(order = order.d[j, ], ar = na.omit(ar.d[j, ]), ma = na.omit(ma.d[j, ])), n, 
                               n.start = n_warm, start.innov = innov_burnin[, j],  
                               innov = innov_insample[, j])[(order.d[j, 2] + 1):(n + order.d[j, 2])]
    
    y <- head(data.bot[, j], -H) # IMPORTANT
    model_path <- Arima(y, order = order.d[j, ], fixed = c(na.omit(ar.d[j, ]), na.omit(ma.d[j, ]), 0))
    
    
    sample_paths <- sapply(seq(npaths), function(ipath){
      #id <- seq((ipath - 1) * H + 1, ipath * H)
      #simulate(model_path, future = TRUE, nsim = H, innov = innov_future[id, j])
      simulate(model_path, future = TRUE, nsim = H, innov = innov_future[ipath, j, ])
    }, simplify = "array")                         
    list_paths[[j]] <-  sample_paths                        
    
  }

  spaths <- simplify2array(list_paths)
  
  return(list(data.bot, order.gen, spaths))
}

ordergenCor <- ordergenCom <- ordergenNoise <-  orderGen 
data.genCom <- data.genCor <- data.genNoise <- dataGen


simulate_large_hts <- function(n, obj.path){
  
  npaths <- obj.path$npaths
  
  #nodes <- list(6, rep(4, 6), rep(4, 24), rep(4, 96), rep(4, 384))
  #nodes <- list(6, rep(4, 6), rep(4, 24), rep(4, 96))
  nodes <- list(6, rep(4, 6), rep(4, 24))
  
  gmat <- hts:::GmatrixH(nodes)
  gmat <- apply(gmat, 1, table)
  n.tot <- sum(unlist(nodes)) + 1
  n.bot <- sum(nodes[[length(nodes)]])
  


  # Generating data for the common pattern
  varCom <- 0.005
####  
  obj.genCom <- data.genCom(n, 1, var = varCom, npaths)
  
  dataCom <- matrix(obj.genCom[[1]], ncol = 1) # ONE DIMENSIONAL
  allCom <- dataCom[, rep(1, times = n.bot)] 
  # Only 2 series out of 4 at the bottom level contains the common pattern
  idxCom <- c(seq(1, n.bot, 4), seq(2, n.bot, 4))
  allCom[, idxCom] <- 0
  
  
  allCom.path <- sapply(seq(npaths), function(ipath){
    datComP <- matrix(obj.genCom[[3]][, ipath, 1], ncol = 1)
    allComP <- datComP[, rep(1, times = n.bot)]
    allComP[, idxCom] <- 0
    allComP
  }, simplify = "array")
  
  # Adding noise to the common pattern
  varRange <- list(0.4, 0.4, 0.4, 0.4, 0.4) # variance of the noise for each level (level 1 to level 5)
  
  # Generating data for the correlated pattern
  bCor <- runif(n.bot/4, 0.3, 0.8) # block correlations
  bCorMat <- lapply(bCor, function(x){
    y <- matrix(x, 4, 4)
    diag(y) = 1
    return(y)
  })
  
  corMatB <- bdiag(bCorMat) # correlation matrix at the bottom level
  varVec <- runif(n.bot, 0.05, 0.1)
  varMat <- as.matrix(Diagonal(x = sqrt(varVec)) %*% corMatB %*% Diagonal(x = sqrt(varVec))) # cov matrix
#### 
  
  obj.genCor <- data.genCor(n, n.bot, varMat, npaths)
  allCor <- obj.genCor[[1]] # generates data with correlated errors
  
  allCor.path <- obj.genCor[[3]]
  

  if(FALSE){  
    
  # generates white noise errors for level 1 to level 4 (level 5: bottom level ignored)
  noiseL <- noiseL.path <- list()
  for(i in 1:(length(nodes)-1))
  {
    nodesLv <- sum(nodes[[i]])
    nLv <- nodesLv / 2
    var.mat <- diag(nLv)
    diag(var.mat) <- rep(varRange[[i]], nLv)
####     
    datL <- rmvnorm(n, rep(0, nLv), var.mat)
    datL <- datL[, rep(1:ncol(datL), each = 2)] # replicating to get the data for -ve part
    sign.vec <- rep(c(1, -1), nLv) # adding +/- into the data
    datL <- t(t(datL) * sign.vec / as.numeric(gmat[[i+1]])) # contribution that passes to the bottom level
    datL <- datL[, rep(1:ncol(datL), times = gmat[[i+1]])] # all noise series at the bottom level
    noiseL[[i]] <- datL
####  
    #datL.path <- rmvnorm(n, rep(0, nLv), var.mat)
    noiseL.path[[i]] <- sapply(seq(npaths), function(ipath){
      datLpath <- rmvnorm(H, rep(0, nLv), var.mat)
      datLpath <- datLpath[, rep(1:ncol(datLpath), each = 2)] # replicating to get the data for -ve part
      datLpath <- t(t(datLpath) * sign.vec / as.numeric(gmat[[i+1]])) # contribution that passes to the bottom level
      datLpath <- datLpath[, rep(1:ncol(datLpath), times = gmat[[i+1]])] # all noise series at the bottom level
      datLpath
    }, simplify = "array")
    
  }
  
  
  
  # generate ARMA series for the noise at the bottom level
  var.mat <- diag(n.bot/2)
  diag(var.mat) <- rep(varRange[[length(nodes)]], sum(nodes[[length(nodes)]])/2)
####   
  obj.genNoise <- data.genNoise(n, n.bot/2, var.mat, npaths)
  
  noiseB <- obj.genNoise[[1]]
  noiseB <- noiseB[, rep(1:ncol(noiseB), each = 2)]
  sign.vec <- rep(c(1, -1), n.bot/2)
  noiseB <- t(t(noiseB) * sign.vec) 
  noiseB[, idxCom] <- 0L # adding noise only to common component
  
  noiseB.path <- sapply(seq(npaths), function(ipath){
    noiseBp <- obj.genNoise[[3]][, ipath, ]
    noiseBp <- noiseBp[, rep(1:ncol(noiseBp), each = 2)]
    sign.vec <- rep(c(1, -1), n.bot/2)
    noiseBp <- t(t(noiseBp) * sign.vec) 
    noiseBp[, idxCom] <- 0L
    noiseBp
    }, simplify = "array")
  
  }# IF FALSE
  
  # common + correlated + noise
  if(FALSE){
    allB <- allCom + allCor + Reduce("+", noiseL) + noiseB
    
    allB.path <- sapply(seq(npaths), function(ipath){
      allCom.path[, , ipath] + allCor.path[, ipath, ] + Reduce("+", noiseL.path)[, , ipath] + noiseB.path[, , ipath]
    }, simplify = "array")
    sample_paths <- allB.path
  }else if(TRUE){
    allB <- allCom + allCor
    allB.path <- sapply(seq(npaths), function(ipath){
      allCom.path[, , ipath] + allCor.path[, ipath, ]
    }, simplify = "array")
    sample_paths <- allB.path
    
    #browser()
  }else if(FALSE){
    
    varCom <- 0.005
    obj.genCom <- data.genCom(n, 1, var = varCom, npaths)
    dataCom <- matrix(obj.genCom[[1]], ncol = 1) # ONE DIMENSIONAL
    allCom <- dataCom[, rep(1, times = n.bot)] 
    # Only 2 series out of 4 at the bottom level contains the common pattern
    idxCom <- c(seq(1, n.bot, 4), seq(2, n.bot, 4))
    #allCom[, idxCom] <- 0
    allCom[, idxCom] <- rnorm(nrow(allCom))
    allCom.path <- sapply(seq(npaths), function(ipath){
      datComP <- matrix(obj.genCom[[3]][, ipath, 1], ncol = 1)
      allComP <- datComP[, rep(1, times = n.bot)]
      #allComP[, idxCom] <- 0
      allComP[, idxCom] <- rnorm(nrow(allComP))
      allComP
    }, simplify = "array")
    
    allB <- allCom 
    allB.path <- sapply(seq(npaths), function(ipath){
      allCom.path[, , ipath] 
    }, simplify = "array")
    sample_paths <- allB.path
  }
  
  

  
  dat.new <- ts(allB, frequency = 1L)
  sim.hts <- hts(dat.new, nodes = nodes)
  #ally <- allts(sim.hts)
  S <- smatrix(sim.hts)
  A <- head(S, nrow(S) - ncol(S))
  
  list(A = A, bts = allB, sim.hts = sim.hts, sample_paths = sample_paths)
}

simulate_small_hts <- function(n_simul, marg = NULL, obj.path = NULL){
  p <- 2; d <- 0; q <- 1
  
  ar_param <- ma_param <- vector("list", 4)
  for(j in seq(4)){
    if(fixed.dgp){
      phi_2 <- 0.5327375 #runif(1, min = 0.5, max = 0.7)
      phi_1 <- -0.2739356 #runif(1, min = phi_2 - 1, max = 1 - phi_2)
      theta_1 <-  0.5469495 #runif(1, min = 0.5, max = 0.7)
    }else{
      phi_2 <- runif(1, min = 0.5, max = 0.7)
      # !!!!! cause non-stationarity runif(1, min = phi_2 - 1, max = 1 - phi_2) 
      # ROB EXPLANATION: if not using true parameters, there is a bias and close to boundary, makes it wordse
      # phi_1 <- runif(1, min = phi_2 - 0.9, max = 0.9 - phi_2) 
      phi_1 <- runif(1, min = phi_2 - 0.71, max = 0.71 - phi_2)
      theta_1 <- runif(1, min = 0.5, max = 0.7)
    }
    ar_param[[j]] <- c(phi_1, phi_2); 
    ma_param[[j]] <- c(theta_1)
  }
    
  mean_process <- 0
  mus <- rep(0, 4)
  #a <- 3.3; 
  #Sigma <-  rbind(c(a, 3, 2, 1), c(3, a, 2, 1), c(2, 2, a, 3), c(1, 1, 3, a))
  varVec <- rep(1, 4) #varVec <- rep(2, 4)
  
  corMatB <- rbind(c(1, 0.7, 0.2, 0.3), 
                   c(0.7, 1, 0.3, 0.2),
                   c(0.2, 0.3, 1, 0.6),
                   c(0.3, 0.2, 0.6, 1))
  # corMatB <- diag(4)
  
  
  Sigma <- as.matrix(Diagonal(x = sqrt(varVec)) %*% corMatB %*% Diagonal(x = sqrt(varVec)))
  
  innov_insample <- gen.innov(n_simul, marg = marg, mus, Sigma)
  innov_burnin   <- gen.innov(n_warm, marg = marg, mus, Sigma)
  
  #n.start <- p + q
  #start.innov <- rep(0, n.start)
  
  bts <- sapply(seq(4), function(j){
    #arima.sim(n = n_simul, list(order = c(p, d, q), ar = ar_param, ma = ma_param), 
    #          n.start = n.start, start.innov = start.innov, innov = innov_insample[, j])
    arima.sim(n = n_simul, list(order = c(p, d, q), ar = ar_param[[j]], ma = ma_param[[j]]), 
              n.start = n_warm, start.innov = innov_burnin[, j], innov = innov_insample[, j])
  })
  
  if(!is.null(obj.path)){
    stopifnot(T_test == H)
    
    npaths <- obj.path$npaths
    
    models_path <- lapply(seq(4), function(j){
      y <- head(bts[, j], -H) # IMPORTANT !!!
      Arima(y, order = c(p, d, q), fixed = c(ar_param[[j]], ma_param[[j]], mean_process))
    })
    
    innov_future <-  gen.innov(npaths * H, marg = marg, mus, Sigma)
    sample_paths <- sapply(seq(npaths), function(ipath){
      #print(ipath)
      #innov_future <-  gen.innov(H, marg = marg)
      id <- seq((ipath - 1) * H + 1, ipath * H)
      sapply(seq(4), function(j){
        # simulate(models_path[[j]], future = TRUE, nsim = H, innov = innov_future[, j])
        simulate(models_path[[j]], future = TRUE, nsim = H, innov = innov_future[id, j])
      })
    }, simplify = "array")
    #innov_future <- mvrnorm(n = H, mus, Sigma = Sigma)
    #paths <- t(replicate(npaths, simulate(model, future = TRUE, nsim = H, innov = innov_future[, j])))
    
  }else{
    sample_paths <- NULL
  }
  
  list(bts = bts, param = list(ar_param = ar_param, ma_param = ma_param), sample_paths = sample_paths)
}




