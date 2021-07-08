# This function computes the permutations needed by aggregation.R to restore the dependences between the variables of interest.
make_permutations <- function(my_bights, residuals){
  
  itree <- my_bights$itree
  mat_residuals <- residuals
  n_resid <- nrow(mat_residuals)
  
  # compute the parsing order of the aggregate nodes
  leaves <- V(itree)[degree(itree, mode="out") == 0]
  agg_nodes <- V(itree)[degree(itree, mode="out") != 0]
  all_nodes <- V(itree)
  
  stopifnot(all(names(all_nodes) == colnames(residuals)))
  
  list_matpermutations <- list_vecties <- vector("list", length(agg_nodes))
  
  for(inode in seq_along(agg_nodes)){
    agg_node <- agg_nodes[inode]
    idseries_agg <- names(agg_node)
    children_nodes <- ego(itree, order = 1, nodes = agg_node, mode = "out")[[1]][-1]

    idchildren <- match(children_nodes, all_nodes)
    mat_residuals <- residuals[, idchildren]

    if (identical(NULL, ncol(mat_residuals))) {
      vec_ties <- 0.0
    } else {
      vec_ties <- sapply(seq(ncol(mat_residuals)), function(j){
        (nrow(mat_residuals) - length(unique(mat_residuals[, j]))) / nrow(mat_residuals)
      }) * 100
    }

    mat_residuals <- tail(mat_residuals, M)
    if (identical(NULL, ncol(mat_residuals))) {
      # browser()
      mat_permutations <- rank(mat_residuals, ties.method="random")
      dim(mat_permutations) <- c(length(mat_permutations), 1)
    } else {
      mat_permutations <- apply(mat_residuals, 2, rank, ties.method = "random")
    }
    colnames(mat_permutations) <- names(children_nodes)
    
    list_matpermutations[[inode]] <- mat_permutations
    list_vecties[[inode]] <- vec_ties
  }
  list_matpermutations <- setNames(list_matpermutations, names(agg_nodes))
  list_vecties         <- setNames(list_vecties, names(agg_nodes))
  list(list_matpermutations = list_matpermutations, list_vecties = list_vecties)
}




