* Install all missing packages: install.package('whatever')

* Problem installing igraph:

    * https://github.com/igraph/rigraph/issues/135
    * This worked:
        Minor update, brew unlink suite-sparse was sufficient for me to get igraph to install (then ran brew link suite-sparse to quickly re-enable).

* Problem install copula, gsl:

    * https://stackoverflow.com/questions/24781125/installing-r-gsl-package-on-mac
    * This worked: brew install gsl; but we first need to do: 
    
        * git -C /usr/local/Homebrew/Library/Taps/homebrew/homebrew-core fetch --unshallow)
        * git -C /usr/local/Homebrew/Library/Taps/homebrew/homebrew-cask fetch --unshallow
        * Then brew install gsl
        * In R console: install.packages('gsl')

* Problem while running source("MinT_ecov.R")

    * First complaint: lubridate not found
        * https://stackoverflow.com/questions/34887837/problems-in-using-lubridate-library
        * install.packages("stringi")
        * install.packages("lubridate") 
        
    * no data files! Because your shell scripts failed silently!
    
        * do source("basef.R") then you realize you have to install several packages.
        * install.packages("fBasics")
        * install.packages("msm")
        * install.packages("gtools")
                
* Common errors:

    * you always have to create folder structure yourself before running scripts!
    
        * mkdir work/basef/KD-IC-NML/
        * mkdir work/basef/

* Step 1:

    * Run this script by removing & to make sure you wait until it ends.
        . ./run_basef.sh
        
    *
        
        
        
        
* Running on Ubuntu:

    * Installing R: https://linuxize.com/post/how-to-install-r-on-ubuntu-20-04/
    * install.packages("fBasics")
    * install.packages("msm")
    * install.packages("gtools") 
    * install.packages("lubridate")
    * install.packages("scoringRules")
    * on cmd: sudo apt-get install libcurl4-openssl-dev 
    * install.packages("forecast")
    * install.packages("abind")
    * install.packages("glmnet")    
    
    * The shell script ran (two times) successfully after that: ./run_basef.sh
    * For MinT_ecov.R, strangely some files were missing
        * residuals_MINT_7938_KD-IC-NML.Rdata: I had to rerun run_basef.sh by finding the right meter index: 36
        * residuals_MINT_8403_KD-IC-NML.Rdata: right meter index: 
        * Rerunning with fewer number of parallel jobs solved it.
        
    * source("makebf_byhalfour.R") is failing because revised forecasts are not there!
    * I had to uncomment save in aggregation.R   
    
     
