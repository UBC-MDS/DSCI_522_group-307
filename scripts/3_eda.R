# author: Reiko Okamoto
# date: 2020-01-23

# Code attribution: Tiffany Timbers' breast-cancer-predictor EDA script

"Creates EDA tables and figures for the pre-processed training set
from the Adult dataset from the UCI Machine Learning Repository 
(https://archive.ics.uci.edu/ml/datasets/Adult).
The tables and figures are saved as .csv and .png files, respectively.

Usage: scripts/3_eda.R --train=<train> --out_dir=<out_dir> --out_dir_plt=<out_dir_plt>

Options:  
--train=<train>             Path to cleaned training data (as feather file)
--out_dir=<out_dir>         Path to directory where tables should be saved
--out_dir_plt=<out_dir_plt> Path to directory where plot should be saved
" -> doc

library(feather)
library(tidyverse)
library(plyr)
library(docopt)
library(ggridges)
library(ggthemes)
theme_set(theme_minimal())

opt <- docopt(doc)

main <- function(train, out_dir, out_dir_plt) {
  
  # LOAD DATASET
  df <- read_feather(train)
  
  cat_feat <- c("workclass", "education", "marital_status", "occupation", "relationship", "race", "sex", "native_country")
  num_feat <- c("age", "education_num", "capital_gain", "capital_loss", "hours_per_week")
  
  # CREATE TABLE TO SUMMARIZE NUMERICAL FEATURES
  num_df <- df %>%
    select(num_feat)
  
  num_mean <- num_df %>%
    apply(2, mean)
  num_min <- num_df %>%
    apply(2, min)
  num_max <- num_df %>%
    apply(2, max)
  
  num_table <- data.frame(num_mean, num_min, num_max) %>%
    dplyr::rename(mean = num_mean, 
                  min = num_min, 
                  max = num_max) %>%
    mutate(feature = num_feat) %>%
    select(feature, mean, min, max)
  
  # WRITE TABLE
  write_csv(num_table, paste0(out_dir, "/num_feat_summary.csv"))
  
  # CREATE TABLE TO SUMMARIZE CATEGORICAL FEATURES
  cat_df <- df %>%
    select(cat_feat)
  
  n_uniq_feat <- c()
  uniq_feat <- c()
  for (feat in cat_feat) {
    print(feat)
    uniq <- unique(cat_df[[feat]])
    uniq_feat <- append(uniq_feat, paste(uniq, collapse = ", "))
    print(uniq_feat)
    n_uniq <- length(uniq)
    n_uniq_feat <- append(n_uniq_feat, n_uniq)
  }
  cat_table <- data.frame(feature = cat_feat, n_uniq_feat, uniq_feat)
  
  # WRITE TABLE 
  write_csv(cat_table, paste0(out_dir, "/cat_feat_summary.csv"))
  
  # CREATE PLOT TO SUMMARIZE DISTRIBUTION OF NUMERICAL FEATURES
  num_plot <- df %>%
    select(num_feat, target) %>%
    gather(key = feature, value = value, -target) %>%
    ggplot(aes(x = value, y = target, color = target, fill = target)) +
    facet_wrap(. ~ feature, scale = "free") +
    geom_density_ridges(alpha = 0.8) +
    scale_fill_few() +
    scale_color_few() +
    guides(fill = FALSE, color = FALSE) +
    theme(axis.title.x = element_blank(),
          axis.title.y = element_blank())
  
  # WRITE PLOT
  ggsave(paste0(out_dir_plt, "/dist_num_feat.png"),
         plot = num_plot,
         width = 6,
         height = 9)
}

main(opt[["--train"]], opt[["--out_dir"]], opt[["--out_dir_plt"]])
