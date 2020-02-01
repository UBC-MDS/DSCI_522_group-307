# author: Reiko Okamoto
# date: 2020-01-31

# Code attribution: Tiffany Timbers' breast-cancer-predictor EDA script

"Creates EDA figures using the pre-processed training set
from the Adult dataset from the UCI Machine Learning Repository 
(https://archive.ics.uci.edu/ml/datasets/Adult).
The figures are saved as a PNG file.

Usage: scripts/3_eda.R --train=<train> --out_dir=<out_dir>

Options:  
--train=<train>             Path to cleaned training data (as feather file)
--out_dir=<out_dir>         Path to directory where figures should be saved
" -> doc

library(feather)
library(tidyverse)
library(docopt)
library(ggthemes)
library(testthat)
library(gridExtra)
theme_set(theme_minimal())

opt <- docopt(doc)

#' Creates EDA figures 
#'
#' @param train path to cleaned training data
#' @param out_dir path to directory where figures should be saved
#'
#' @return PNG file
#' 
#' @example 
#' main("data/clean_train_data.feather", "results")
main <- function(train, out_dir) {
  
  # LOAD DATASET
  df <- read_feather(train)
  
  # CREATE PLOT TO SUMMARIZE DISTRIBUTION OF `age`
  p1 <- df %>%
    ggplot(aes(age)) +
    geom_histogram(aes(fill = target), color = "white", bins = 30) +
    facet_grid(target ~ .) +
    labs(x = "Age",
         y = "Count") +
    theme_bw() +
    theme(axis.text = element_text(size = 16),
          axis.title = element_text(size = 16),
          strip.text = element_text(size = 14),
          legend.position = "none")
  
  # CREATE PLOT TO SUMMARIZE DISTRIBUTION OF `education_num`
  p2 <- df %>%
    ggplot(aes(education_num)) +
    geom_histogram(aes(fill = target), color = "white", bins = 17) +
    facet_grid(target ~ .) +
    labs(x = "Educational attainment (years)",
         y = "Count") +
    theme_bw() +
    theme(axis.text = element_text(size = 16),
          axis.title = element_text(size = 16),
          strip.text = element_text(size = 14),
          legend.position = "none",
          axis.title.y = element_blank())
  
  # CREATE PLOT TO SUMMARIZE DISTRIBUTION OF `hours_per_week`
  p3 <- df %>%
    ggplot(aes(hours_per_week)) +
    geom_histogram(aes(fill = target), color = "white", bins = 22) +
    facet_grid(target ~ .) +
    labs(x = "Working hours per week",
         y = "Count") +
    theme_bw() +
    theme(axis.text = element_text(size = 16),
          axis.title = element_text(size = 16),
          strip.text = element_text(size = 14),
          legend.position = "none",
          axis.title.y = element_blank())
  
  num_plot <- grid.arrange(p1, p2, p3, ncol = 3)
  
  # WRITE PLOT
  ggsave(paste0(out_dir, "/numerical.png"),
         plot = num_plot,
         width = 12,
         height = 6)
}

test_main <- function() {
  test_that("Input should be a feather file", {
    expect_equal(str_sub(opt[["--train"]], -7), "feather")
  })
}

test_main()

main(opt[["--train"]], opt[["--out_dir"]])
