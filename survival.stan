/*
Survival analysis model coded as a Poisson model with 1 = event and 0 = no event.
Regressors are time-varying. Baseline hazard is assumed to be Weibull.
Based on Tomi Peltola's code at http://becs.aalto.fi/en/research/bayes/diabcvd/wei_hs.stan
*/

data {
  int<lower=0> Nobs;  // number of data points
  int<lower=0> M;  // number of regressors
  int<lower=0> event[Nobs];  // 0 = no event, 1 = event
  vector[Nobs] time;  // time in trial
  matrix[Nobs, M] X;  // regressor matrix
  real<lower=0> dt;  // time bin size
  int<lower=1> Ntypes;  // number of trial types
  int<lower=1> ttype[Nobs];  // trial type index
}

transformed data {
  real<lower=0> tau_mu;
  real<lower=0> tau_al;

  tau_mu = 10.0;
  tau_al = 10.0;
}

parameters {
  // parameters used to make a half-Cauchy tau_s
  real<lower=0> tau_global;
  
  // parameters used to make local half-Cauchy tau_j's
  vector<lower=0>[M] tau_local;
  
  // parameters for hazard functions
  vector[Ntypes] mm;  // mean
  vector<lower=0>[Ntypes] ss;  // standard deviation
  
  // regression coefficents
  vector[M] beta_raw;

  real mu;  // beta0 parameter
}

transformed parameters {
  vector[M] beta;
  vector[Nobs] hazard;

  beta = tau_global * tau_local .* beta_raw;
  
  for (idx in 1:Nobs) {
    hazard[idx] = exp(normal_lpdf(time[idx] | mm[ttype[idx]], ss[ttype[idx]]))/(1 - Phi_approx((time[idx] - mm[ttype[idx]]) / ss[ttype[idx]]));
  }
}

model {
  event ~ poisson(dt * hazard .* exp(mu + X * beta));

  mm ~ normal(3.0, 3.0);
  ss ~ cauchy(0, 3);
  
  beta_raw ~ normal(0.0, 1.0);
  
  tau_global ~ cauchy(0, 1);
  tau_local ~ cauchy(0, 1);

  mu ~ normal(0.0, tau_mu);  
}
