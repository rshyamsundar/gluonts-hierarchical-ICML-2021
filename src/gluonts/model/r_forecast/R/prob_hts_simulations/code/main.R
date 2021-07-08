# This is the main script which computes the base forecasts as well as all the revised forecasts from the other methods.
# The arguments are:
# experiment: the experiment you want to perform ("small", "newlarge")
# marg: the distribution of the errors ("norm", "t")
# T_learn: the size of the learning set (training + validation)
# M: the number of observations per samples (when sampling from the predictive distributions)
# idjob: the job id (useful when using distributed computing, see runsim.sh)
# nb_simulations: the nnumber of simulations to perform
# nbcores: the number of cores used by the job
# use.trueparam: do you want to use the true AR parameters? (TRUE, FALSE)
rm(list = ls())
assign("last.warning", NULL, envir = baseenv())
args = (commandArgs(TRUE))
if(length(args) == 0){
  experiment <- "small"
  marg <- "norm"
  T_learn <- 500
  M <- 500
  idjob <- 1986
  nb_simulations <- 2
  nbcores <- 1
  use.trueparam <- TRUE
}else{
  
  for(i in 1:length(args)){
    print(args[[i]])
  }
  
  experiment <- args[[1]]
  marg <- args[[2]]
  T_learn <- as.integer(args[[3]])
  M <- as.integer(args[[4]])
  use.trueparam <- as.logical(args[[5]])
  idjob <- as.integer(args[[6]])
  nb_simulations <- as.integer(args[[7]])
  nbcores <- as.integer(args[[8]])
}

set.seed(idjob)

graphics.off()

source("config_paths.R")
source("packages.R")
source("bights.R")
source("simulate.R")
source("utils.R")
source("mint.R")
source("code.R")
source("permutations.R")
source("aggregation.R")
source("nicefigs.R")

tag <- "jasa"
do.recovery <- TRUE

fixed.dgp <- FALSE
do.bootstrap.residuals <- TRUE
do.correction <- FALSE

print(fixed.dgp)
print(use.trueparam)
print(do.bootstrap.residuals)
print(do.correction)

if(use.trueparam){
  print("True parameters are used at the bottom and the top levels !")
}

print(experiment)
print(marg)  
print(M)

q_probs <- runif(M)
n_true_paths <- M
n_forecast_paths <- M

H <- 2
mc.cores.basef <- nbcores
refit_step <- 40
T_test <- ifelse(do.recovery, H, 500)
T_all   <- T_learn + T_test
n_warm  <- 500
n_simul <- T_all

print(T_learn)

obj.path <- NULL 
if(do.recovery){
    obj.path <- list(npaths = n_true_paths)
}

config_forecast_bot <- list(fit_fct = auto.arima, refit_fct = Arima, 
                            param_fit_fct = list(seasonal = FALSE, stationary = TRUE, approximation = FALSE, ic = "bic"),  # max.p = 2, max.q = 2, 
                            param_refit_fct = list(use.initial.values = TRUE),
                            param_forecast = list( bootstrap = do.bootstrap.residuals, npaths = n_forecast_paths ))

config_forecast_agg <- list(fit_fct = auto.arima, refit_fct = Arima, 
                            param_fit_fct = list(seasonal = FALSE, stationary = TRUE, approximation = FALSE, ic = "bic"), 
                            param_refit_fct = list(use.initial.values = TRUE),
                            param_forecast = list( bootstrap = do.bootstrap.residuals, npaths = n_forecast_paths ))

list_simulations  <- vector("list", nb_simulations)
for(i in seq_along(list_simulations)){
  
  print(paste(i, " - Start ALL -", base::date(), sep = ""))
  
  if(experiment == "newlarge"){
    obj_simul <- list()
    obj_simul$param <- NULL
    obj_simul$param$ar_param <- obj_simul$param$ma_param <- NULL
    
    ngroups <- 25
    list_params <- vector("list", ngroups)
    
    res <- NULL
    for(igroup in seq(ngroups)){
      NM <- cbind(paste("T", igroup, sep = "") , 
                  c( paste("A", igroup, sep = "") , paste("B", igroup, sep = "") , 
                     paste("C", igroup, sep = "") , paste("D", igroup, sep = "")   ))
      res <- rbind(res, NM)
    }
    res <- cbind("T", res)
    
    tags <- cbind(res[, 1], sapply(seq(2, ncol(res)), function(j)(
      apply(res[, seq(j)], 1, paste, collapse = "")
    )))
    
    myinfo <- makeINFO2(tags)
    A <- myinfo$A
    
    bts <- matrix(NA, nrow = n_simul, ncol = ngroups * 4)
    sample_paths_bottom <- array(NA, c(H, ngroups * 4, obj.path$npaths))
    for(igroup in seq(ngroups)){
      res_sim <- simulate_small_hts(n_simul, marg, obj.path)
      obj_simul$param$ar_param <- c(obj_simul$param$ar_param, res_sim$param$ar_param)
      obj_simul$param$ma_param <- c(obj_simul$param$ma_param, res_sim$param$ma_param)
      
      id <- seq((igroup - 1) * 4 + 1, (igroup - 1) * 4 + 4 )
      bts[, id] <- res_sim$bts
      sample_paths_bottom[, id, ] <- res_sim$sample_paths
    }
    
  }else if(experiment == "small"){
    obj_simul <- simulate_small_hts(n_simul, marg, obj.path)
    bts <- obj_simul$bts
    
    NM <- cbind(rep("T", 4), rep(c("A", "B"), each = 2), rep(c("A", "B"), 2))
    tags <- cbind(NM[, 1], sapply(seq(2, ncol(NM)), function(j)(
      apply(NM[, seq(j)], 1, paste, collapse = "")
    )))
    
    myinfo <- makeINFO(tags)
    A <- myinfo$A
    sample_paths_bottom <- obj_simul$sample_paths
    
  }
  ###########
  my_bights <- bights(bts, A)
  my_bights$itree <- myinfo$itree
  my_bights$bnames <-  myinfo$bottomSeries
  my_bights$anames <-  myinfo$aggSeries
  my_bights$allnames <- c(myinfo$aggSeries, myinfo$bottomSeries)
  
  
  sample_paths <- sapply(seq(H), function(h){
    as.matrix(my_bights$S %*% sample_paths_bottom[h, , ])
  }, simplify = 'array')
  
  print(paste("Data simulated: ", " ", base::date()))
  
  P_BU <- pbu(my_bights)
  
  infofile <- file.path(rdata.folder, paste("info_", experiment, ".Rdata", sep = ""))
  save(file =  infofile, list = c("my_bights"))

  list_subsets_test <- lapply(seq(T_learn, T_all - H), function(i){c(i - T_learn + 1, i)})
  
  
  results <- makebf(my_bights, list_subsets_test, H = H, 
                    config_forecast_agg = config_forecast_agg, config_forecast_bot = config_forecast_bot, 
                    refit_step = refit_step, mc.cores = mc.cores.basef)
  
  e_hat <- results$allresiduals
  row.names(e_hat) <-  my_bights$allnames
  e_hat <- e_hat[, sample(seq(ncol(e_hat)), M)]
  
  mint_methods <- list(mintdiagonal = "diagonal", mintshrink = "shrink")
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
  obj_permutations_ehat <- make_permutations(my_bights, t(e_hat))
  obj_permutations_etilde <- make_permutations(my_bights, t(e_tilde))
  
  
  n_test <- length(list_subsets_test)
  test_depbu_samples <- test_base_samples <- test_mint_samples <- test_indepbu_samples <- vector("list", n_test)
  test_results <- vector("list", n_test)
  for(itest in seq(n_test)){
    
    if(itest %% 100 == 0)
      print(itest)
    
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
    base_samples <- sapply(seq(my_bights$nts), function(i){
      sapply(seq(H), function(h){
        #qf <- results$allqf[[i]][[itest]][, h]
        qf <- results$allqf[[i]]$list_qf[[itest]][, h]
        sample(qf)
      })
    }, simplify = "array")
    base_samples <- aperm(base_samples, c(1, 3, 2))
    
    test_results[[itest]]$base <- base_samples
    
    # DEP-BU AND INDEP-BU
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
    
    test_results[[itest]]$depbu <- depbu_samples
    test_results[[itest]]$depbumint <- depbumint_samples
    test_results[[itest]]$indepbu <- indepbu_samples
  
  }# test
  
  
  
  if(do.recovery){
      list_simulations[[i]] <- list(samples_pred = test_results[[1]], samples_true = sample_paths)
  }else{
    err_test <- lapply(seq(H), function(h){
      lapply(seq(n_test), function(itest){ # itest
        lapply(test_results[[itest]], function(mat_method){ # imethod
          obs <- results$allfuture[, itest, h]
          X <- mat_method[, , h]
          
          crps <- compute_crps(X, obs)
          
          squared_erors <- (apply(X, 2, mean) - obs)^2
          
          qs <- compute_qscores(X, obs)
          qs_tails <- apply(qs, 2, function(x){ mean(x * weights_tails) })
          qs_uniform <- apply(qs, 2, function(x){ mean(x * weights_uniform) })
          qs_rtail <- apply(qs, 2, function(x){ mean(x * weights_rtail) })
          qs_ltail <- apply(qs, 2, function(x){ mean(x * weights_ltail) })
          
          # DO NOT SAVE qs (too big)
          list(crps = crps, squared_erors = squared_erors, qs_tails = qs_tails, qs_rtail = qs_rtail, qs_ltail = qs_ltail, qs_uniform = qs_uniform)
        })
      })
    })
    list_simulations[[i]] <- err_test
    
  }
    if(i%%5 == 0){
      filetosave <- nameResfile(experiment, marg, T_learn, M, use.trueparam, idjob, tag)
      save(file = filetosave, list = c("list_simulations"))
    }
  
}
  filetosave <- nameResfile(experiment, marg, T_learn, M, use.trueparam, idjob, tag)
  save(file = filetosave, list = c("list_simulations"))
