There was a problem installing `gsl`, a dependency for running `PERMBU_MINT` method.
Here is some help resolving this issue on mac and ubuntu.

* Problem installing gsl:
    
    * On mac:
        * https://stackoverflow.com/questions/24781125/installing-r-gsl-package-on-mac
        * This worked:  
    
            * git -C /usr/local/Homebrew/Library/Taps/homebrew/homebrew-core fetch --unshallow
            * git -C /usr/local/Homebrew/Library/Taps/homebrew/homebrew-cask fetch --unshallow
            * brew install gsl
            * R -e 'install.packages(c("gsl"))'
            
    * On Ubuntu:
        * sudo apt install libgsl-dev          
        * R -e 'install.packages(c("gsl"))'              
