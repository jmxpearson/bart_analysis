# load up useful vars
source('helpers.R')

# load analysis output
load(file=paste(ddir, 'lfpfitdata', sep='/'))

# now plot an ROC curve for a given subject
ind <- 10  # numbered consecutively, starting at 1

# get performance dataframe
perf <- get_performance(ind)

# plot to file
pdf(file='roc.pdf', paper='USr', width=11, height=8.5)
plotroc(perf)
dev.off()

######## code to plot heatmap of regression coefficients ##########
df <- extract_coeffs(fitobjs[[ind]])

# now reorder channels based on hierarchical clustering
coef_grid <- spread(df, band, value)
dend <- hclust(sign_neutral_dist(coef_grid[, -1]))

# useful summary stats
band_stats <- df %>% group_by(band) %>%
    summarise(mean=mean(value), mean_abs=mean(abs(value)), std=sd(value),
              std_abs=sd(abs(value)))

pdf(file='lfp_coeff_grid.pdf', paper='USr', width=11, height=8.5)
plt <- plot_lfp_coefficient_grid(df)
print(plt)
dev.off()
