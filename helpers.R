# helper functions for R analysis
source('setup_env.R')

# function to calculate roc curve
roc <- function(fitobj, y, X, dth=0.01) {
  # make a dataframe of true positive, false positive measures for a
  # variety of thresholds
  thvec <- seq(0, 1, dth)
  N <- length(thvec)
  TPR <- rep(NA, N)
  FPR <- rep(NA, N)
  for (ind in 1:N) {
    pp <- predict(fitobj, X, type='response') > thvec[ind]
    TP <- sum(pp == TRUE & y == TRUE)
    FP <- sum(pp == TRUE & y == FALSE)
    TN <- sum(pp == FALSE & y == FALSE)
    FN <- sum(pp == FALSE & y == TRUE)
    TPR[ind] <- TP/(TP + FN)
    FPR[ind] <- FP/(FP + TN)
  }
  roc <- data.frame(thresh=thvec, TPR=TPR, FPR=FPR)
  return (roc)
}

# function to return dataframe suitable for plotting roc curve
get_performance <- function(ind) {
    glmobj <- fitobjs[[ind]]$glmobj
    fname <- paste(paste(chanlist[ind,], collapse='.'), 'lfpglmdata.csv', sep='.')
    dfile <- paste(ddir, fname, sep='/')
    dat <- read.csv(dfile, header=TRUE, row.names=1, colClasses=c('numeric'))

    # get performance
    min.ind <- which(glmobj$lambda == glmobj$lambda.min)
    auc <- glmobj$cvm[min.ind]

    # make predictions, calculate ROC
    y <- dat[,1]
    X <- as.matrix(dat[,-1])
    perf <- roc(glmobj, y, X, dth=0.001)
    return (list(perf=perf, auc=auc))
} 

# plot an roc curve
plotroc <- function(perf) {
    titlestr = paste('Area Under the Curve =', sprintf('%.2f', perf$auc))
    
    plt <- ggplot(perf$perf, aes(FPR, TPR))
    plt + geom_line(size=1.5) + geom_abline(slope=1, intercept=0) +
      labs(title = titlestr, x='\nFalse Positive Rate', y='True Positive Rate\n') +
      scale_x_continuous(limits=c(0,1), expand=c(0,0)) +
      scale_y_continuous(limits=c(0,1), expand=c(0,0)) + theme_bw() +
      theme(plot.title = element_text(lineheight=1, size=36), 
            axis.text=element_text(color='black', size=12),
            axis.title.x=element_text(size=20), axis.title.y=element_text(size=20)
            )
}

# make a raster plot of a time series
rasterize <- function(df, an) {
    # df is a dataframe to constitute the raster (time, trial)
    # an is a (trial, time) dataframe of points to annotate

    jet.colors <- colorRampPalette(c("#00007F", "blue", "#007FFF", "cyan", "#7FFF7F", "yellow", "#FF7F00", "red", "#7F0000"))

    titlestr = 'Event-related power\n'
    plt <- ggplot(df, aes(x=as.numeric(time), y=as.numeric(trial), 
        fill=as.numeric(value)))

    pp <- plt + geom_raster() + 
    scale_fill_gradientn(colours=jet.colors(7), limits=c(-80, 0)) + 
    geom_vline(xintercept=0, show_guide=FALSE, size=1) + 
    labs(title = titlestr, x='\nTime (s)', y='Trial\n') + 
    scale_x_continuous(expand=c(0,0)) + 
    scale_y_continuous(expand=c(0,0)) + 
    theme_bw() + 
    theme(plot.title = element_text(lineheight=1, size=28), axis.text=element_text(color='black', size=12), axis.title.x=element_text(size=20), axis.title.y=element_text(size=20) ) +
    annotate('point', x=an[, 2], y=an[, 1], color='black', size=2)

    return(pp)
}

chanmeans <- function(df) {
    # plot mean values for power in each channel
    plt <- ggplot(df, aes(as.numeric(time), as.numeric(value)))

    pp <- plt + 
    geom_line(aes(color=as.factor(variable))) + 
    scale_colour_hue(h=c(90, 180), guide=FALSE) + 
    labs(x='\nTime From Stopping (s)', y='Normalized Channel Power\n') +
    theme_bw() + 
    theme(plot.title = element_text(lineheight=1, size=36), axis.text=element_text(color='black', size=12),
        axis.title.x=element_text(size=20), axis.title.y=element_text(size=20))

    return(pp)
}

extract_coeffs <- function(fitobj) {
  coeffs <- melt(fitobj$beta)
  splitnames <- unlist(lapply(rownames(coeffs), 
    FUN=function(x){strsplit(x, "[.]")}))
  newcols <- matrix(splitnames, ncol=2, byrow=TRUE)
  colnames(newcols) <- c("band", "channel")
  df <- cbind(coeffs, newcols)
  cmat <- cast(df, channel ~ band)
  df <- melt(cmat)
  df$band <- factor(df$band, levels=c('delta', 'theta', 'alpha', 
      'beta', 'gamma'))
  rownames(df) <- c()
  return(df)
}

sign_neutral_dist <- function(M) {
    # compute a distance function between rows of M, allowing 
    # for an overall sign flip to one of them
    # this preserves the correlation within rows, but says that sign
    # changes between rows are immaterial
    M <- as.matrix(M)
    n <- nrow(M)

    dminus <- matrix(NA, n, n)
    for (i in 1:n) {
        for (j in 1:(i - 1)) {
            dminus[i, j] <- sum((M[i,] - M[j,])^2)
        }
    }

    dplus <- matrix(NA, n, n)
    for (i in 1:n) {
        for (j in 1:(i - 1)) {
            dplus[i, j] <- sum((M[i,] + M[j,])^2)
        }
    }

    dfinal = pmin(dplus, dminus)
    return(as.dist(dfinal))
}

plot_lfp_coefficient_grid <- function(df) {
  # plot a heatmap grid of coefficients for lfp regression
  plt <- ggplot(df, aes(x=band, y=channel))
  pp <- plt + geom_tile(aes(fill=abs(value)), color='gray') +
    scale_fill_gradient2(low='blue', high='red', na.value='white', midpoint=0,
        guide=guide_colorbar(title='Absolute\nregression\ncoefficient')) +
    scale_x_discrete('Frequency Band', expand=c(0, 0)) +
    scale_y_discrete('Channel', expand=c(0, 0)) +
    theme(axis.ticks=element_blank(), axis.text=element_text(color='black',
        size=12), axis.title.x=element_text(size=20), 
        axis.title.y=element_text(size=20))

  return(pp)
}

plot_spike_coefficient_grid <- function(df) {
  # plot a heatmap grid of coefficients for spike count regression
  plt <- ggplot(df, aes(x=unit, y=variable))
  pp <- plt + geom_tile(aes(fill=value), color='gray') +
    scale_fill_gradient2(low='blue', high='red', na.value='white', 
      midpoint=0, 
      guide=guide_colorbar(title='Percent change\nfrom baseline'), 
      limits=c(-50, 50)) +
    scale_x_discrete('Unit', expand=c(0, 0), limits=unique(df$unit),
      breaks=seq(5, max(unique(df$unit)), 5)) +
    scale_y_discrete('Regressor', expand=c(0, 0)) +
    theme(axis.ticks=element_blank(), axis.text=element_text(color='black',
        size=12), axis.title.x=element_text(size=20), 
        axis.title.y=element_text(size=20))

  return(pp)
}