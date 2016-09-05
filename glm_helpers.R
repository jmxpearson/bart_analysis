fit_all_and_save <- function(filext, outname, family, datalist, measure, lambdatype, shuffle=FALSE, nfolds=10) {
  fitobjs <- list()
  for (ind in 1:dim(datalist)[1]) {
    fname <- paste(paste(datalist[ind,], collapse='.'), filext, sep='.')
    dfile <- paste(ddir, fname, sep='/')
    print(dfile)
    dat <- read.table(dfile, sep=',', header=TRUE, row.names=1, colClasses=c('numeric'))
    thisfit <- run_glm(dat, family, measure, lambdatype, shuffle, nfolds)
    fitobjs[[ind]] <- thisfit
  }

  save(fitobjs, file=paste(ddir, outname, sep='/'))
}

run_glm <- function(dframe, type='binomial', measure="deviance", lambdatype='1se', shuffle=FALSE, nfolds=10){
  # given an input data frame, perform an elastic net regression
  # on the data contained therein

  # make model matrix
  y <- dframe[, 1]
  X <- dframe[, -1]

  if (shuffle) y <- sample(y)

  allobjs = list()
  alphalist = seq(0.1, 1, 0.1)

  folds <- partition_data(y, nfolds=nfolds)

  # loop over alpha values
  for (alpha in alphalist) {
    # run sparse regression
    # don't include constant in X; glmnet will fit an intercept,
    # but if we supply
    # one manually, it will be penalized, which we don't want
    glmobj <- cv.glmnet(data.matrix(X), y, alpha = alpha, family = type,
                        foldid = folds, intercept = TRUE, type.measure = measure)

    # pull out regression coefficients for best model using different
    # regularization strategies
    for (reg in lambdatype) {
      allobjs[[reg]][[length(allobjs[[reg]]) + 1]] <- get_best_beta(glmobj, reg)
    }
  }

  bestobj <- list()
  for (reg in lambdatype) {
    if (reg == 'none') {
      # if no regularization, pick smallest alpha
      bestobj[[reg]] <- allobjs[[reg]][[1]]
    } else {
      bestobj[[reg]] <- get_best_alpha(allobjs[[reg]], alphalist)
    }
  }

  # print auc curve to figure
  svg('auc.svg')
  # use first listed regularization type
  plot(bestobj[[lambdatype[1]]]$glmobj)
  dev.off()

  # if user only asked for one lambda type, return single object, not list
  # of objects
  if (length(bestobj) == 1) bestobj <- bestobj[[1]]

  return(bestobj)
}

partition_data <- function(outcome, nfolds) {
  suppressMessages(library('caret'))

  folds <- createFolds(outcome, nfolds)

  foldid <- rep(NA, length(outcome))

  for (ind in 1:nfolds) {
    foldid[folds[[ind]]] <- ind
  }

  return(foldid)
}

get_best_beta <- function(glmobj, lambdatype) {
  # given a glmobj from elastic net, extract the cross-validation score and
  # coefficients of the best fit

  if (lambdatype == '1se') { minlambda <- glmobj$lambda.1se }
  else if (lambdatype == 'min') { minlambda <- glmobj$lambda.min }
  else if (lambdatype == 'none') { minlambda <- min(glmobj$lambda) }

  minlambda.ind <- which(glmobj$lambda == minlambda)
  fit <- glmobj$glmnet.fit
  beta <- fit$beta[, minlambda.ind]
  intercept <- fit$a0[minlambda.ind]
  score <- glmobj$cvm[minlambda.ind]

  return(list(beta = beta, intercept = intercept, score = score, glmobj = glmobj))
}

get_best_alpha <- function(objlist, alphalist) {
  # given a list of objects returned by get_best_beta and a list of alpha
  # values, return the object corresponding to the highest score
  measure_type <- tolower(objlist[[1]]$glmobj$name)
  if (grepl('auc', measure_type)) {
    extract_best <- which.max
  } else {
    extract_best <- which.min
  }

  scorelist <- sapply(objlist, function(x) {x$score})
  best_ind <- extract_best(scorelist)  # if using auc, want max score

  bestobj <- c(objlist[[best_ind]], alpha=alphalist[best_ind])

  return(bestobj)
}
