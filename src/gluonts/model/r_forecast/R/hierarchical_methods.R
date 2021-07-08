library("hts")
library(parallel)

naive_bottom_up <- function(hts, params) {
    h <- params$prediction_length
    fmethod <- params$fmethod
    fcasts1.bu <- forecast(
      hts,
      h = h,
      method="bu",
      fmethod = fmethod,
      parallel = TRUE,
    )
    aggts(fcasts1.bu)
}

mint <- function(hts, params) {
    h <- params$prediction_length
    fmethod <- params$fmethod
    covariance <- params$covariance
    nonnegative <- params$nonnegative
    algorithm <- params$algorithm

    print(c("covariance:", covariance))

    ally <- aggts(hts)
    n <- nrow(ally)
    p <- ncol(ally)

    allf <- matrix(NA, nrow = h, ncol = p)

    if (tolower(covariance) != "ols") {
        res <- matrix(NA, nrow = n, ncol = p)
    }

    for(i in 1:p)
    {
      fit <- if (fmethod == "arima") auto.arima(ally[, i]) else ets(ally[, i])
      allf[, i] <- forecast(fit, h = h)$mean
      if (tolower(covariance) != "ols") {
          res[, i] <- na.omit(ally[, i] - fitted(fit))
      }
    }

    if (tolower(covariance) == "ols") {
        # OLS is a special case of MinT where the covariance matrix is identity.
        if (is.hts(hts)){
          y.f_cg <- combinef(allf, get_nodes(hts), weights = NULL,
                             keep = "all", algorithms = algorithm, nonnegative = nonnegative, parallel = TRUE)
        } else{
          y.f_cg <- combinef(allf, groups = get_groups(hts), weights = NULL,
                             keep = "all", algorithms = algorithm, nonnegative = nonnegative, parallel = TRUE)
        }
        return(y.f_cg)  # mean forecasts
    } else {
        if (is.hts(hts)){
          y.f_cg <- MinT(allf, get_nodes(hts), residual = res, covariance = covariance,
                         keep = "all", algorithms = algorithm, nonnegative = nonnegative, parallel = TRUE)
        } else{
          y.f_cg <- MinT(allf, groups = get_groups(hts), residual = res, covariance = covariance,
                         keep = "all", algorithms = algorithm, nonnegative = nonnegative, parallel = TRUE)
        }
        return(y.f_cg)  # mean forecasts
    }
}

################################
############## ERM #############
################################
# compute ERM matrix
erm_matrix <- function(S, Y, Y_hat){
  # computes erm matrix
  # output : P : bottom time series rows x totaltime series

  S     <-as.matrix(S)
  Y     <-as.matrix(Y)
  Y_hat <-as.matrix(Y_hat)


  n <- dim(Y_hat)[2]

  temp1 <- solve(t(S)%*%S)
  temp2 <- (t(S)%*%t(Y))%*%Y_hat
  temp3 <- solve(t(Y_hat)%*%Y_hat + 1e-3*diag(n))
  P     <- (temp1%*%temp2)%*%temp3

  return(P)
}

##################################

get_base_forecasts <- function(ally, h, fmethod){

  n <- nrow(ally)
  p <- ncol(ally)

  allf <- matrix(NA, nrow = h, ncol = p)

  for(i in 1:p)
  {
    fit <- if (fmethod == "arima") auto.arima(ally[, i]) else ets(ally[, i])
    allf[, i] <- forecast(fit, h = h)$mean
  }

  return(allf[nrow(allf),])
}

##################################
###############################
ermParallel <- function(hts, params){
  # input
  # ally: data matrix - each column is a time series, each row is an observation in time
  # params: params$prediction_length : prediction length
  #       : params$context_length    : context_length [WARNING: this parameter is
  #                                    specific to this method and is related to
  #                                    the parameter T1 from the paper]
  #       : params$S                 : Summation matrix
  #       : params$fmethod           : method used for base forecasts: "arima" or "ets"
  ally <-  aggts(hts)
  n <- nrow(ally)
  p <- ncol(ally)

  H  <- params$prediction_length
  T1 <- n - (params$prediction_length+1)

  print(c("n:", n))
  print(c("p:", p))
  print(c("T1:", T1))
  print(c("T-H:", dim(ally)[1]-H))

  S  <- smatrix(hts)
  fmethod <- params$fmethod

  # check types
  S    <- as.matrix(S)
  ally <- as.matrix(ally)

  #
  T    <- dim(ally)[1]
  allf <- matrix(NA, nrow = H, ncol = p)

  numCores <- detectCores()
  Yhat_reconciled <- mclapply(seq(1,H), erm_aux, T1=T1, T=T, p=p, ally=ally, fmethod=fmethod, S=S, mc.cores = numCores)

  for(h in seq(1,H)){
    allf[h,] <- Yhat_reconciled[[h]]
  }

  return(allf)

}


##################################

erm_reconcile_forecasts <- function(P, S, Y_hat){

  Y_hat <- t(as.matrix(Y_hat))
  Y_recon <- t((S%*%P)%*%t(Y_hat))

  return(Y_recon)
}

################################
erm <- function(hts, params){
  # input
  # ally: data matrix - each column is a time series, each row is an observation in time
  # params: params$prediction_length : prediction length
  #       : params$context_length    : context_length [WARNING: this parameter is
  #                                    specific to this method and is related to
  #                                    the parameter T1 from the paper]
  #       : params$S                 : Summation matrix
  #       : params$fmethod           : method used for base forecasts: "arima" or "ets"

  ally <-  aggts(hts)
  n <- nrow(ally)
  p <- ncol(ally)

  H  <- params$prediction_length
  T1 <- n - (params$prediction_length+1)

  print(c("n:", n))
  print(c("p:", p))
  print(c("T1:", T1))
  print(c("T-H:", dim(ally)[1]-H))

  S  <- smatrix(hts)
  fmethod <- params$fmethod

  # check types
  S    <- as.matrix(S)
  ally <- as.matrix(ally)

  #
  T    <- dim(ally)[1]
  allf <- matrix(NA, nrow = H, ncol = p)

  for(h in seq(1,H)){
    print(c("h:", h))
    Y      <- ally[seq(T1+h,T),]
    Yhat_h <- matrix(NA, nrow = T-T1-h+1, ncol = p)

    c <- 1 #counter
    for(i in seq(T1,T-h)){
      Yin        <- ally[seq(1,i),]
      Yhat_h[c,] <- get_base_forecasts(Yin, h, fmethod)
      c          <- c + 1
    }

    P    <- erm_matrix(S, Y, Yhat_h)

    Yhat <- get_base_forecasts(ally, h, fmethod)
    Yhat_reconciled <- erm_reconcile_forecasts(P, S, Yhat)

    allf[h,] <- Yhat_reconciled
  }

  return(allf)

}

erm_aux <- function(h, T1, T, p, ally, fmethod, S){

  print(c("h:", h))
  Y      <- ally[seq(T1+h,T),]
  Yhat_h <- matrix(NA, nrow = T-T1-h+1, ncol = p)

  c <- 1 #counter
  for(i in seq(T1,T-h)){
    Yin        <- ally[seq(1,i),]
    Yhat_h[c,] <- get_base_forecasts(Yin, h, fmethod)
    c          <- c + 1
  }

  P    <- erm_matrix(S, Y, Yhat_h)

  Yhat <- get_base_forecasts(ally, h, fmethod)
  Yhat_reconciled <- erm_reconcile_forecasts(P, S, Yhat)

  return(Yhat_reconciled)


}

depbu_mint <- function(bts, tags, params) {
    print(getwd())
    # setwd("./prob_hts_simulations/code/")
    # install.packages("here")
    work_path <- "src/gluonts/model/r_forecast/R/prob_hts_simulations/code"

    source(here::here(work_path, "config_paths.R"))
    source(here::here(work_path, "packages.R"))
    source(here::here(work_path, "bights.R"))
    source(here::here(work_path, "simulate.R"))
    source(here::here(work_path, "utils.R"))
    source(here::here(work_path, "mint.R"))
    source(here::here(work_path, "code.R"))
    source(here::here(work_path, "permutations.R"))
    source(here::here(work_path, "aggregation.R"))

    freq_period <<- params$freq_period
    H <<- params$prediction_length
    M <<- params$num_samples
    n_forecast_paths <<- M
    T_learn <<- params$train_length
    do.correction <<- params$correction
    seasonal <- params$seasonal
    stationary <- params$stationary

    # NM <- cbind(rep("T", 4), rep(c("A", "B"), each = 2), rep(c("C", "D"), 2))
    # tags <- cbind(NM[, 1], sapply(seq(2, ncol(NM)), function(j)(
    #   apply(NM[, seq(j)], 1, paste, collapse = "")
    # )))

    mc.cores.basef <- 1
    refit_step <- 40
    do.bootstrap.residuals <<- TRUE
    use.trueparam <<- FALSE

    q_probs <- runif(M)

    # Create the hierarchical representation
    #print(tags)
    myinfo <- makeINFO2(tags)
    A <- myinfo$A
    my_bights <- bights(bts, A)
    my_bights$itree <- myinfo$itree
    my_bights$bnames <-  myinfo$bottomSeries
    my_bights$anames <-  myinfo$aggSeries
    my_bights$allnames <- c(myinfo$aggSeries, myinfo$bottomSeries)

    Mhalf = M / 2
    config_forecast_bot <- list(fit_fct = auto.arima, refit_fct = Arima,
                            param_fit_fct = list(seasonal = seasonal, stationary = stationary, approximation = FALSE, ic = "bic"),  # max.p = 2, max.q = 2,
                            param_refit_fct = list(use.initial.values = TRUE),
                            param_forecast = list( bootstrap = do.bootstrap.residuals, level = seq(1, Mhalf)/(Mhalf+1) ))

    config_forecast_agg <- list(fit_fct = auto.arima, refit_fct = Arima,
                            param_fit_fct = list(seasonal = seasonal, stationary = stationary, approximation = FALSE, ic = "bic"),
                            param_refit_fct = list(use.initial.values = TRUE),
                            param_forecast = list( bootstrap = do.bootstrap.residuals, level = seq(1, Mhalf)/(Mhalf+1) ))

    # config_forecast_bot <- list(fit_fct = auto.arima, refit_fct = Arima,
    #                         param_fit_fct = list(seasonal = seasonal, stationary = stationary, approximation = FALSE, ic = "bic"),  # max.p = 2, max.q = 2,
    #                         param_refit_fct = list(use.initial.values = TRUE),
    #                         param_forecast = list( bootstrap = do.bootstrap.residuals, npaths = n_forecast_paths ))
    #
    # config_forecast_agg <- list(fit_fct = auto.arima, refit_fct = Arima,
    #                         param_fit_fct = list(seasonal = seasonal, stationary = stationary, approximation = FALSE, ic = "bic"),
    #                         param_refit_fct = list(use.initial.values = TRUE),
    #                         param_forecast = list( bootstrap = do.bootstrap.residuals, npaths = n_forecast_paths ))

    P_BU <- pbu(my_bights)

    # We just do one multi-step forecast instead of rolling forecasts.
    list_subsets_test <- lapply(seq(T_learn, T_learn), function(i){c(i - T_learn + 1, i)})
    print("Training target's start and final index: ")
    #print(list_subsets_test)

    results <- makebf(my_bights, list_subsets_test, H = H,
                    config_forecast_agg = config_forecast_agg, config_forecast_bot = config_forecast_bot,
                    refit_step = refit_step, mc.cores = mc.cores.basef)

    # print(results)
    print("computing residuals")
    e_hat <- results$allresiduals
    row.names(e_hat) <-  my_bights$allnames
    e_hat <- e_hat[, sample(seq(ncol(e_hat)), M, replace = TRUE)]  # TODO: needed for tourism-small!
    # e_hat <- e_hat[, sample(seq(ncol(e_hat)), M)]

    # browser()
    # mint_methods <- list(mintdiagonal = "diagonal", mintshrink = "shrink")
    mint_methods <- list(mintshrink = "shrink")
    list_MINT <- lapply(mint_methods, function(wmethod){
        compute_pmint(my_bights, method = wmethod, residuals = e_hat)
    })

    # MINT
    res_allmint <- lapply(list_MINT, function(mylist){
        res_obj <- mylist
        means_mint <- sapply(seq(H), function(h){
          as.matrix(my_bights$S %*% res_obj$P_MINT %*%  results$allmf[, , h])
        }, simplify = "array")
        variances_mint <- diag(res_obj$V)
        list(means_mint = means_mint, variances_mint = variances_mint)
    })

    e_tilde <- my_bights$S %*% list_MINT$mintshrink$P_MINT %*% e_hat

    # Make permutations
    print("making permutations")
    obj_permutations_ehat <- make_permutations(my_bights, t(e_hat))
    obj_permutations_etilde <- make_permutations(my_bights, t(e_tilde))

    n_test <- length(list_subsets_test)
    test_depbu_samples <- test_base_samples <- test_mint_samples <- test_indepbu_samples <- vector("list", n_test)
    test_results <- vector("list", n_test)
    for(itest in seq(n_test)){

        samples_almint <- lapply(res_allmint, function(res_mint){
          mint_samples <- sapply(seq(my_bights$nts), function(j){
            sapply(seq(H), function(h){
              m <- res_mint$means_mint[j, itest, h]
              v <- res_mint$variances_mint[j]
              qnorm(q_probs, m, sqrt(v))
            })
           }, simplify = "array")
          mint_samples <- aperm(mint_samples, c(1, 3, 2))
        })

        test_results[[itest]] <- c(test_results[[itest]], samples_almint)

        # sample from all basef
        print("base samples")
        base_samples <- sapply(seq(my_bights$nts), function(i){
          #print(i)
          #print(results$allmf[i, , ])
          sapply(seq(H), function(h){
              # print(h)
              # print(results$allmf[i, , ][h])
              # qf <- results$allqf[[i]]$list_qf[[itest]][, h]
              qf <- results$allqf[[i]][[itest]][, h]
              # sample(qf)
              # browser()
              qf
              # results$allmf[i, , ][h]
              # sample(results$allmf[i, , ][h], size=M, replace=TRUE)
          })
        }, simplify = "array")
        # browser()
        # if (H==1) {
        #     dim(base_samples) <- c(1, 1, length(base_samples))
        # }
        aperm(base_samples, c(1, 3, 2))
        base_samples <- aperm(base_samples, c(1, 3, 2))

        test_results[[itest]]$base <- base_samples

        # browser()
        # DEP-BU AND INDEP-BU
        print("prediction:")
        depbu_samples <- indepbu_samples <- depbumint_samples <- array(NA, dim(base_samples))
        for(h in seq(H)){
          samples_h <- base_samples[, , h]

          # DEP-BU
          permuted_samples_bottom <- permutate_samples(my_bights, samples_h, obj_permutations_ehat$list_matpermutations) # ATTENTION ICI
          tpermuted_samples_bottom <- t(permuted_samples_bottom)
          depbu_samples[, , h] <- as.matrix(t(my_bights$S %*% tpermuted_samples_bottom))

          if(do.correction){
            permuted_samples_bottom <- permutate_samples(my_bights, samples_h, obj_permutations_etilde$list_matpermutations)
            tpermuted_samples_bottom <- t(permuted_samples_bottom)
          }

          # DEP-BU + MINT
          btilde <- P_BU %*% res_allmint[["mintshrink"]]$means_mint[, itest, h]
          meanrevised_samples_bottom <- tpermuted_samples_bottom - apply(permuted_samples_bottom, 2, mean) + btilde
          if(do.correction){
            meanrevised_samples_bottom <- (meanrevised_samples_bottom/apply(permuted_samples_bottom, 2, sd) ) * sqrt(P_BU %*% res_allmint[["mintshrink"]]$variances_mint)
          }
          depbumint_samples[, , h] <- as.matrix(t(my_bights$S %*% meanrevised_samples_bottom))

          # INDEP-BU
          indepbu_samples[, , h] <- as.matrix(t(my_bights$S %*% P_BU %*% t(samples_h)))
        }
    }
    # print(depbumint_samples)
    # depbumint_samples
    #print(DIM(base_samples))
    #print(DIM(results$allmf))
    # browser()
    # aperm(base_samples, c(1, 3, 2))
    aperm(depbumint_samples, c(1, 3, 2))
    # results$allmf
}