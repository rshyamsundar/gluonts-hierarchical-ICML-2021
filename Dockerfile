FROM continuumio/miniconda3:4.7.12

RUN apt-get --allow-releaseinfo-change update && \
    apt-get install -y build-essential

ADD ./environment.yml ./environment.yml

RUN conda install -n base -c conda-forge mamba && \
    mamba env update -n base -f ./environment.yml && \
    conda clean -afy

RUN R -e 'install.packages(c("here", "SGL", "matrixcalc", "igraph", "sn", "scoringRules", "fBasics", "msm", "gtools", "lubridate", "forecast", "abind", "glmnet", "SuppDists"), repos="http://cran.us.r-project.org")'

RUN R -e 'install.packages("hts", repos="http://cran.us.r-project.org")'

RUN R -e 'install.packages(c("gsl", "copula", "propagate", "plotrix"), repos="http://cran.us.r-project.org")'

COPY . .
RUN pip install -e .
