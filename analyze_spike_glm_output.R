# load up useful vars
source('helpers.R')

# load analysis output
load(file=paste(ddir, 'spkfitdata', sep='/'))

######## code to plot heatmap of regression coefficients ##########
if (exists('lambdatype') == FALSE) lambdatype <- 1

pullbetas <- function(x) {
    # grab betas from each unit's fit object

    # if object is a fitobject and not a list of objects
    # (i.e., if only one lambda type was fit)
    if ('beta' %in% names(x)) {
        df <- data.frame(t(x$beta))
    } else {
        # else get the requested lambda type
        df <- data.frame(t(x[[lambdatype]]$beta))
    }
}

betas <- do.call(bind_rows, lapply(fitobjs, FUN=pullbetas))
effects <- exp(betas) * 100 - 100
effects <- cbind(data.frame(unit=1:dim(effects)[1]), effects)
df <- gather(effects, variable, value, -unit)

plt <- plot_spike_coefficient_grid(df)
print(plt)
