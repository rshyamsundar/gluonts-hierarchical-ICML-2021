# This script produces the figures with all the accuracy measures.
rm(list = ls())
source("config_paths.R")
source("utils.R")
source("nicefigs.R")
source("analysis_utils.R")
library(igraph)
library(kSamples)
library(car)
graphics.off()

probs <- seq(1, 99)/100

# task can be: "pvalue"  "mse" "wasserstein"
task <- "wasserstein"
# experiment can be: "small" "newlarge"
#experiment <- "newlarge"
experiment <- "small"

# T_learn <- 10000
# M <- 1000
T_learn <- 500
M <- 500
marg <- "norm"
#marg <- "t"
use.trueparam <- FALSE #TRUE
#use.trueparam <- TRUE
tag <- "jasa"
set_jobs <- seq(1, 12) 

h <- 1
all_simulations <- NULL
for(idjob in set_jobs){
  print(idjob)
  filetoload <- nameResfile(experiment, marg, T_learn, M, use.trueparam, idjob, tag)
  load(filetoload)
  all_simulations <- c(all_simulations, list_simulations)
}

idnull <- which(sapply(all_simulations, is.null))
if(length(idnull) > 0){
  all_simulations <- all_simulations[-which(sapply(all_simulations, is.null))]
}

nb_simulations <- length(all_simulations)
print(nb_simulations)


load(file.path(rdata.folder, paste("info_", experiment, ".Rdata", sep = "")))
nts <- my_bights$nts
naggts <- my_bights$naggts
node_names <- names(V(my_bights$itree))


for(i in seq_along(all_simulations)){
  x_mintshrink <- all_simulations[[i]]$samples_pred$mintshrink
  x_indepbu    <- all_simulations[[i]]$samples_pred$indepbu
  x_indepbumintshrink    <- x_indepbu
  
  mean_mintshrink <- apply(x_mintshrink, c(2, 3), mean)
  mean_indepbu    <- apply(x_indepbu, c(2, 3), mean)
  
  nvar <- dim(x_indepbumintshrink)[1]
  for(k in seq(nvar)){
    x_indepbumintshrink[k, , ] <- x_indepbu[k, , ] - mean_indepbu + mean_mintshrink
  }
  
  all_simulations[[i]]$samples_pred$indepbumintshrink <- x_indepbumintshrink
}


mycols <- c("red", "green" , "grey", "blue", "purple",  "yellow", "orange")
# "mintdiagonal"      "mintshrink"  "base"   "depbu"   "depbumint"  "indepbu"   "indepbumintshrink"

name_methods <- c(names(all_simulations[[1]]$samples_pred), "true")

id1 <- id2 <- seq(M)

id_print <- c(1, 2)
#id_print <- c(1, 2, 30)
#id_print <- c(1, 2, 30, 40, 50, 60)

mat_test <- sapply(id_print, function(j){
  if(j%%10 == 0)
  print(j)
  
  sapply(seq(nb_simulations), function(i){
    
    samples_pred <- all_simulations[[i]]$samples_pred
    samples_true <- all_simulations[[i]]$samples_true
    samples_true <- aperm(samples_true, c(2, 1, 3))
    
    samples_method <- sapply(samples_pred, function(mat){
      mat[, j, h]
    })

    x2 <- samples_true[, j, h]
    if(task == "mse"){
      mu <- mean(x2)
      RET <- sapply(colnames(samples_method), function(mycol){
        x1 <- samples_method[, mycol]
        (mean(x1) - mu)^2
      })
    }else if(task == "pvalue"){
      RET <- sapply(colnames(samples_method), function(mycol){
        x1 <- samples_method[, mycol]
        ks.test(x1[id1], x2[id2])$p.value
      })
    }else if(task == "wasserstein"){
      true_quantiles <- quantile(x2, probs)
      RET <- sapply(colnames(samples_method), function(mycol){
        x1 <- samples_method[, mycol]
        log(mean( (quantile(x1, probs) - true_quantiles)^2))
      })
    }
    RET
  })
}, simplify = "array")

mat_test <- aperm(mat_test, c(2, 1, 3))
print(dimnames(mat_test))


######### BOXPLOTS (pvalues, mse, wasserstein)
  myfile <- namePdffile(experiment, marg, T_learn, M, use.trueparam,  
                        paste(tag, "-", task,sep = "") )
  
  if(T_learn == 500){
    savepdf(myfile, width = 21 * 0.9, height = 29.7 * 0.25)
    par(mfrow  = c(1, 3), mar=c(5.05,3,2,2))
  }else{
    savepdf(myfile, width = 21 * 0.9, height = 29.7 * 0.20)
    par(mfrow  = c(1, 3), mar=c(1,3,2,2))
  }
  
  
  for(j in seq(dim(mat_test)[3])){
    mymat <- mat_test[, , j]
    
    if(T_learn == 500){
      colnames(mymat) <- sapply(colnames(mymat), bettername)
      myxaxt <- NULL
    }else{
      myxaxt <- 'n'
    }
    
    if(task != "mse"){
      if(j == 1){
        mymain <- "Level 1 (top level)"
      }else if(j == 3){
        mymain <- "Level 3 (bottom level)"
      }else{
        mymain <- "Level 2"
        #paste("Level ", j, ifelse(j == 3, "(bottom level)", "") )
      }
      
      
      if(task == "pvalue"){
        boxplot(mymat, las=2, ylab = "p-value", main = mymain, 
                col = mycols, yaxt = 'n', xaxt = myxaxt, cex.axis = 1, outline = FALSE)
        axis(2, at = c(0, 0.25, 0.5, 0.75, 1), cex.axis = 1.2)
        abline(h = c(0, 0.25, 0.5, 0.75, 1), lwd = 0.1)
      }else{
        boxplot(mymat, las=2, ylab = "2-Wasserstein distance (log scale)", 
                main = mymain, col = mycols, xaxt = myxaxt, cex.axis = 1, outline = FALSE)
      }
      
    }else{
      boxplot(mymat, las=2, ylab = "MSE", 
              main = paste("Leval ", j, ifelse(j == 3, "(bottom level)", "") ), 
              col = mycols, cex.axis = 1.2,
              outline = FALSE)
    }
  }
  
  dev.off()
