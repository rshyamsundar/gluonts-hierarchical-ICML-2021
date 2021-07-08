
bettername <- function(namemethod, option = 1){
  if(namemethod == "mintdiagonal"){
    #RET <- "NL-MD" 
    RET <- ifelse(option == 1, "Norm- \n MinTDiag", "Norm-MinTDiag")
  }else if(namemethod == "mintshrink"){
    #RET <- "NL-MS"  
    RET <- "Norm- \n MinTShrink"
    RET <- ifelse(option == 1, "Norm- \n MinTShrink", "Norm-MinTShrink")
  }else if(namemethod == "base"){
    RET <- "BASE"
  }else if(namemethod == "depbu"){
    #RET <- "DB-NM"   
    RET <- ifelse(option == 1, "DepBU- \n NoMinT", "DepBU-NoMinT")
  }else if(namemethod == "depbumint"){
    #RET <- "DB-MS"   
    RET <- ifelse(option == 1, "DepBU- \n MinTShrink", "DepBU-MinTShrink")
  }else if(namemethod == "indepbu"){
    #RET <- "IB-NM" 
    RET <- ifelse(option == 1, "IndepBU- \n NoMinT", "IndepBU-NoMinT")
  }else if(namemethod == "indepbumintshrink"){
    #RET <- "IB-MS" 
    RET <- ifelse(option == 1, "IndepBU- \n MinTShrink" , "IndepBU-MinTShrink")
  }else{
    RET <- namemethod
  }
  RET
}