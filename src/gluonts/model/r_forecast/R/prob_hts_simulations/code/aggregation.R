# This function applies the permutations to the samples generated from the marginal distributions.
# In other words, it restores the dependences between the variables.
permutate_samples <- function(my_bights, my_samples, list_matpermutations){
 
  n_permutations <- unique(sapply(list_matpermutations, nrow))
  stopifnot(length(n_permutations) == 1)
  
  stopifnot(nrow(my_samples) == n_permutations)
  
  itree <- my_bights$itree
  aggSeries <- my_bights$anames
  bottomSeries <- my_bights$bnames
  A <- my_bights$A
  ##########
  # compute the parsing order of the aggregate nodes
  leaves <- V(itree)[degree(itree, mode="out") == 0]
  agg_nodes <- V(itree)[degree(itree, mode="out") != 0]
  
  depth_aggnodes <- sapply(agg_nodes, function(agg_node){
    vec <- distances(itree, agg_node, leaves, mode = "out")
    max( vec[which(vec!=Inf)])
  })
  
  ordered_agg_nodes_names <- names(sort(depth_aggnodes))
  ordered_agg_nodes <- V(itree)[match(ordered_agg_nodes_names, V(itree)$name)]
  ##########
  
  base_samples <- my_samples
  
  base_samples_bottom <- my_samples[, seq(my_bights$naggts + 1, my_bights$nts)]
  colnames(base_samples_bottom) <- my_bights$bnames
  
  perm_samples_bottom <- base_samples_bottom
  variables <- colnames(perm_samples_bottom)
  
  mat_test <- NULL
  # PERM-BU
  for(inode in seq_along(ordered_agg_nodes)){
    #print(inode)
    
    agg_node <- ordered_agg_nodes[inode]
    idseries_agg <- names(agg_node)
    iagg <- match(idseries_agg, aggSeries)
    children_nodes <- ego(itree, order = 1, nodes = agg_node, mode = "out")[[1]][-1]
    nkids <- length(children_nodes)
    
    mat_permutations <- list_matpermutations[[idseries_agg]]
    
    ranks_historical <- mat_permutations
    stopifnot(all(colnames(ranks_historical) == names(children_nodes)))
    
    depth_node <- depth_aggnodes[match(idseries_agg, names(depth_aggnodes))]
    
    samples_children <- matrix(NA, nrow = M, ncol = nkids)
    
    columns_agg <- which(children_nodes %in% agg_nodes)
    columns_bottom <- which(children_nodes %in% leaves)
    children_names <- names(children_nodes)
    
    # Extracting/computing the samples for each child
    if(length(columns_agg) > 0){
      id_agg_children  <- match(children_names[columns_agg], aggSeries)
      samples_agg_children <- as.matrix(t(tcrossprod(A[id_agg_children, , drop = F], perm_samples_bottom)))
      samples_children[, columns_agg] <- samples_agg_children
    }
    
    if(length(columns_bottom) > 0){
      id_bottom_children  <- match(children_names[columns_bottom], bottomSeries)
      samples_children[, columns_bottom] <- perm_samples_bottom[, id_bottom_children]
    }
    
    # Computing the ranks of the samples for each child
    ranks_samples_children <- sapply(seq(ncol(samples_children)), function(j){
      rank(samples_children[, j], ties.method = "random")
    })
    
    index_mat <- sapply(seq(nkids), function(j){
      res <- match(ranks_historical[, j], ranks_samples_children[, j])
      stopifnot(all(!is.na(res)))
      res
    })
    
    # Permutating the rows
    if(length(columns_bottom) > 0){
      perm_samples_bottom[, id_bottom_children] <- sapply(seq_along(id_bottom_children), function(j){
        perm_samples_bottom[index_mat[, columns_bottom[j]], id_bottom_children[j]]
      })
    }
    
    if(length(columns_agg) > 0){
      res <- lapply(seq_along(id_agg_children), function(j){
        id <- which(A[id_agg_children[j], ] == 1)
        perm_samples_bottom[index_mat[, columns_agg[j]], id, drop = F]
      })
      ids <- lapply(id_agg_children, function(id_agg_child){
        which(A[id_agg_child, ] == 1)
      })
      ids <- unlist(ids)
      perm_samples_bottom[, ids] <- do.call(cbind, res)
    }
    
  }# agg node

  perm_samples_bottom
}
