# load up useful vars
source('helpers.R')

# load analysis output
load(file=paste(ddir, 'lfpfitdata', sep='/'))

# loop over datasets, extracting summaries
setnames <- paste(chanlist$V1, chanlist$V2, sep='.')
fit_table <- data.frame()
for (ind in seq_along(fitobjs)) {
    fo <- fitobjs[[ind]]
    df <- extract_coeffs(fo)
    df$dataset <- setnames[ind]

    # get summaries
    # number of nonzero entries for each band
    sdf <- df %>% group_by(band) %>%
                  summarise(dataset=max(dataset),
                            nnz=sum(value != 0)) %>%
                  spread(band, nnz)

    # number of channels used
    sdf <- cbind(sdf, df %>% group_by(channel) %>%
                        summarise(sumcoeff=sum(abs(value))) %>%
                        summarise(nchan=sum(sumcoeff != 0)))

    # auc
    glmo <- fo$glmobj
    min.ind <- which(glmo$lambda == glmo$lambda.1se)
    auc <- glmo$cvm[min.ind]
    aucsd <- glmo$cvsd[min.ind]
    sdf$auc <- auc
    sdf$sd <- aucsd

    fit_table <- rbind(fit_table, sdf)
}

write.csv(fit_table, file=paste(ddir, 'lfpfitsummary.csv', sep='/'),
          row.names=FALSE)
