# Dockerfile for the DSCI 522 Project - Income Level Predictor
# Author: Evhen Dytyniak
# Date: 2020-01-07

#Use continuumio anaconda base image 
FROM continuumio/anaconda3@sha256:9068d4877089bf01ab633e7b26571683a90ba8883b5e47bd12d1e3eae27efea3

# install base R and python packages
RUN apt-get update &&\
    apt-get install r-base r-base-dev -y &&\
    pip install docopt==0.6.2 &&\
    pip install feather-format==0.4.0 &&\
    pip install --upgrade pandas==0.25.2 &&\
    pip install --upgrade numpy==1.17.4 &&\
    pip install --upgrade scikit-learn==0.22

# install R packages
RUN apt-get update &&\
    apt install libcurl4-openssl-dev libssl-dev libxml2-dev &&\
    Rscript -e "install.packages('tidyverse', repos = 'https://mran.revolutionanalytics.com/snapshot/2020-01-02')" &&\
    Rscript -e "install.packages('knitr', repos = 'https://mran.revolutionanalytics.com/snapshot/2020-01-01')" &&\
    Rscript -e "install.packages('feather', repos = 'https://mran.revolutionanalytics.com/snapshot/2020-01-01')" &&\
    Rscript -e "install.packages('docopt', repos = 'https://mran.revolutionanalytics.com/snapshot/2020-01-01')" &&\
    Rscript -e "install.packages('ggthemes', repos = 'https://mran.revolutionanalytics.com/snapshot/2020-01-01')" &&\
    Rscript -e "install.packages('testthat', repos = 'https://mran.revolutionanalytics.com/snapshot/2020-01-01')" &&\
    Rscript -e "install.packages('gridExtra', repos = 'https://mran.revolutionanalytics.com/snapshot/2020-01-01')" &&\
    Rscript -e "install.packages('rlang', repos = 'https://mran.revolutionanalytics.com/snapshot/2020-01-01')" &&\
    Rscript -e "install.packages('rmarkdown', repos = 'https://mran.revolutionanalytics.com/snapshot/2020-01-01')" &&\
    Rscript -e "install.packages('kableExtra', repos = 'https://mran.revolutionanalytics.com/snapshot/2020-01-01')"