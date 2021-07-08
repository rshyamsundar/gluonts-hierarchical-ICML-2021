#!/bin/bash

#full_path=$(realpath $0)
#dir_path=$(dirname $full_path)
#mypath=$(dirname $dir_path )

mypath="/Users/rangapur/prob-hts/simulations/"
rscript="main.R"

experiment="small" && marg="norm" && Tlearning=500 && M=500 && Trueparam="FALSE"
#experiment="newlarge" && marg="norm" && Tlearning=500 && M=500 && Trueparam="FALSE"
#experiment="small" && marg="norm" && Tlearning=10000 && M=1000 && Trueparam="FALSE"
#experiment="newlarge" && marg="norm" && Tlearning=10000 && M=1000 && Trueparam="FALSE"

nbsimul=1
nbcores=3
alljobs=$(seq 1 12)

for idjob in ${alljobs[@]}
do
      flag="$experiment-$Tlearning-$idjob"
      echo "$flag"
      Rscript --vanilla $rscript $experiment $marg $Tlearning $M $Trueparam $idjob $nbsimul $nbcores  > "$mypath/work/rout/$flag.Rout" 2> "$mypath/work/rout/$flag.err" &
done
