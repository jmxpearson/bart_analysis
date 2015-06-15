# set up file structure and common variables used by R code

adir <- '.'  # analysis directory
ddir <- 'data'  # data directory

# lfp data
cfile <- 'data/lfp_channel_file.csv'
chanlist <- read.csv(paste(adir, cfile, sep='/'), header=FALSE)
chanlist <- chanlist[, 1:2]
chanlist <- chanlist[!duplicated(chanlist),]

# spike data
cfile <- 'data/valid_units.csv'
unitlist <- read.csv(paste(adir, cfile, sep='/'), header=FALSE)

suppressMessages(library(glmnet))
suppressMessages(library(ggplot2))
suppressMessages(library(plyr))
suppressMessages(library(reshape))
