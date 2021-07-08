# This script defines some functions to generate the base forecasts.
makebf <- function(obj_bights, list_subsets, H, config_forecast_agg, config_forecast_bot, refit_step, mc.cores = 1){
  n <- obj_bights$nts
  m <- obj_bights$nbts
  results <- vector("list", n)

  results <- mclapply(seq(n), function(j){
    if(j <= n-m){
      config_forecast <- config_forecast_agg
    }else{
      config_forecast <- config_forecast_bot
    }

    rolling.forecast(obj_bights$yts[, j], list_subsets, H, config_forecast, refit_step = refit_step, j)
  }, mc.cores = mc.cores)

  allmf <- simplify2array(lapply(results, "[[", "mf"))
  allfuture <- simplify2array(lapply(results, "[[", "future"))

  #allqf <- lapply(results, "[[", "list_qf")
  allqf <- lapply(results, "[[", "list_qf")

  allresiduals <- simplify2array(lapply(results, "[[", "e_residuals"))

  if (H==1) {
    dim(allmf) <- c(1, 1, length(allmf))
    dim(allfuture) <- c(1, 1, length(allfuture))

    for(i in 1:length(allqf)) {
      allqf[[i]][[1]] = aperm(allqf[[i]][[1]], c(2, 1))
    }

  }
  # browser()
  allmf <- aperm(allmf, c(3, 1, 2))
  allfuture <- aperm(allfuture, c(3, 1, 2))
  allresiduals <- t(allresiduals)

  list(allmf = allmf, allfuture = allfuture, allqf = allqf, allresiduals = allresiduals)

}

rolling.forecast <- function(series, list_subsets, H, config_forecast, refit_step, idseries){
  n_subsets <- length(list_subsets)
  mf <- future <- matrix(NA, nrow = n_subsets, ncol = H)
  list_qf <- vector("list", n_subsets)

  fit_fct <- config_forecast$fit_fct
  refit_fct <- config_forecast$refit_fct
  param_fit_fct <- config_forecast$param_fit_fct
  param_refit_fct <- config_forecast$param_refit_fct
  param_forecast <- config_forecast$param_forecast


  for(i in seq(n_subsets)){
    ts_split <- list_subsets[[i]]
    if(is.ts(series)){
      learn_series <- subset(series, start = ts_split[1], end = ts_split[2])
    }else{
      learn_series <- series[seq(ts_split[1], ts_split[2])]
    }

    future[i, ] <- series[ts_split[2] + seq(1, H)]


    if( (i-1) %% refit_step == 0){
      if(use.trueparam && idseries > my_bights$naggts){
          ar_param <- obj_simul$param$ar_param[[idseries - my_bights$naggts]]
          ma_param <- obj_simul$param$ma_param[[idseries - my_bights$naggts]]
          stopifnot(!is.null(ar_param) && !is.null(ma_param))
          model <- Arima(y = learn_series, order = c(length(ar_param), 0, length(ma_param)),
                fixed = c(ar_param, ma_param, 0))
      }else{
        # browser()
        # model <- do.call(fit_fct, c(list(y = learn_series), param_fit_fct))
        model <- ets(ts(learn_series, frequency=freq_period))
      }

    }else{
      # browser()
      # model <- do.call(refit_fct, c(list(y = learn_series, model = model), param_refit_fct))
      model <- ets(ts(learn_series, frequency=freq_period))
    }

    if(i == 1){
      # e_residuals <- as.numeric(resid(model))
      e_residuals <- as.numeric(na.omit(ts(learn_series, frequency=freq_period) - fitted(model)))
    }

    # browser()
    f <- do.call(forecast, c(list(object = model, h = H), param_forecast))
    # f <- forecast(model, h = H)
    mf[i, ] <- f$mean
    # quantf1 <- sapply(seq(H), function(h){
    #   c(rev(f$lower[h, ]), f$upper[h, ])
    # })
    quantf <- t(sapply(1:M, function(n) { simulate(model, H) }))
    # browser()
    list_qf[[i]] <- quantf
    # browser()

    #nsamples <- param_forecast$npaths
    #f <- t(replicate(nsamples, simulate(model, bootstrap = do.bootstrap.residuals, nsim = H, future = T)))
    #mf[i, ] <- apply(f, 2, mean)
    #list_qf[[i]] <- f
  }

  output <- list(future = future, mf = mf, list_qf = list_qf, e_residuals = e_residuals)
}

DIM <- function( ... ){
    args <- list(...)
    lapply( args , function(x) { if( is.null( dim(x) ) )
                                    return( length(x) )
                                 dim(x) } )
}

makeINFO <- function(tags){
  myedges <- data.frame(rbind(cbind(tags[, 1], tags[, 2]),
                              cbind(tags[, 2], tags[, 3])))
  itree <- graph.data.frame(myedges)
  itree <- simplify(itree, remove.loops = F)


  # Compute A - for each agg. node, compute the associated leafs
  all.nodes.names <- V(itree)$name
  agg.nodes.names <- aggSeries <- all.nodes.names[which(degree(itree, V(itree), "out")!=0)]
  n_agg <- length(agg.nodes.names)

  bottomSeries <- tags[, ncol(tags)]
  n_bottom <- ncol(bts)
  A <- matrix(0, nrow = n_agg, ncol = n_bottom)

  for(i in seq_along(agg.nodes.names)){
    agg.node.name <- agg.nodes.names[i]
    reachable <- which(shortest.paths(itree, agg.node.name, mode="out") != Inf)
    terminal.nodes <- reachable[which(degree(itree, reachable, mode="out") == 0)]
    terminal.nodes.names <- all.nodes.names[terminal.nodes]
    ids <- match(terminal.nodes.names, bottomSeries)
    stopifnot(all(!is.na(ids)))
    A[i, ids] <- 1
  }
  output <- list(bottomSeries = bottomSeries, aggSeries = aggSeries, itree = itree,
                 A = A, n_agg = n_agg, n_bottom = n_bottom)
  return(output)
}


makeINFO2 <- function(tags){
  myedges <- data.frame(do.call(rbind, lapply(seq(ncol(tags) - 1), function(j){
    cbind(tags[, j], tags[, j+1])
  })))

  itree <- graph.data.frame(myedges)
  itree <- simplify(itree, remove.loops = F)


  # Compute A - for each agg. node, compute the associated leafs
  all.nodes.names <- V(itree)$name
  agg.nodes.names <- aggSeries <- all.nodes.names[which(degree(itree, V(itree), "out")!=0)]
  n_agg <- length(agg.nodes.names)

  bottomSeries <- tags[, ncol(tags)]
  n_bottom <- length(bottomSeries)
  A <- matrix(0, nrow = n_agg, ncol = n_bottom)

  for(i in seq_along(agg.nodes.names)){
    agg.node.name <- agg.nodes.names[i]
    reachable <- which(shortest.paths(itree, agg.node.name, mode="out") != Inf)
    terminal.nodes <- reachable[which(degree(itree, reachable, mode="out") == 0)]
    terminal.nodes.names <- all.nodes.names[terminal.nodes]
    ids <- match(terminal.nodes.names, bottomSeries)
    stopifnot(all(!is.na(ids)))
    A[i, ids] <- 1
  }
  output <- list(bottomSeries = bottomSeries, aggSeries = aggSeries, itree = itree,
                 A = A, n_agg = n_agg, n_bottom = n_bottom)
  return(output)
}