suppressMessages(library(glmnet))
source('glm_helpers.R')
source('setup_env.R')
set.seed(77654)

args <- commandArgs(TRUE)
shuffle <- FALSE

if (args[1] == 'lfp') {
    print("Performing LFP analysis...")
    filext <- 'lfpglmdata.csv'
    outname <- 'lfpfitdata' 
    family <- 'binomial'
    datalist <- chanlist
    measure <- 'auc'
    lambdatype <- '1se'
} else if (args[1] == 'spikes') {
    print("Performing spike analysis...")
    filext <- 'spkglmdata.csv'
    outname <- 'spkfitdata'
    family <- 'poisson'
    datalist <- unitlist
    measure <- 'deviance'
    lambdatype <- c('1se', 'min', 'none')
}

if (length(args) > 1) {
    if (args[2] == 'shuffle') {
        shuffle <- TRUE
        lambdatype <- '1se'
    }
}

fit_all_and_save(filext, outname, family, datalist, measure, lambdatype, shuffle)