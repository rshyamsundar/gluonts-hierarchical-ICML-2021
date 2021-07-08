# Some useful functions.
nameResfile <- function(experiment, marg, T_learn, M, use.trueparam, idjob, tag){
  file.path(rdata.folder, 
            paste(experiment, "_", marg, "_", T_learn, "_", M, "_", use.trueparam, "_", idjob, "_", tag, ".Rdata", sep = ""))
}

namePdffile <- function(experiment, marg, T_learn, M, use.trueparam, tag){
  file.path(pdf.folder, 
            paste(experiment, "_", marg, "_", T_learn, "_", M, "_",  use.trueparam, "_", tag, sep = ""))
}

pbu <- function(objhts){
  P_BU <- Matrix(0, nrow = objhts$nbts, ncol = objhts$nts, sparse = T)
  P_BU[cbind(seq(objhts$nbts), seq(objhts$nts - objhts$nbts + 1, objhts$nts)) ] <- 1
  P_BU
}

compute_crps <- function(X, obs){
  sapply(seq(ncol(X)), function(j){
    crps_sample(y = obs[j], dat = X[, j]) # from ScoringRules package
  })
}

compute_qscores <- function(X, obs){
  X_sorted <- apply(X, 2, sort)
  sapply(seq(ncol(X)), function(j){
    qf <- X_sorted[, j]
    2 * ((obs[j] <= qf) - q_probs) * (qf - obs[j])      
  })
}