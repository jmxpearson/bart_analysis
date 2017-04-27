/*
Survival analysis model coded as a Poisson model with 1 = event and 0 = no event.
Regressors are time-varying. Baseline hazard is assumed to be Weibull.
Based on Tomi Peltola's code at http://becs.aalto.fi/en/research/bayes/diabcvd/wei_hs.stan
*/
functions {
  vector sqrt_vec(vector x) {
    vector[dims(x)[1]] res;

    for (m in 1:dims(x)[1]){
      res[m] = sqrt(x[m]);
    }

    return res;
  }

  vector hs_prior_lp(real r1_global, real r2_global, vector r1_local, vector r2_local) {
    // a Cauchy is a t(df = 1) variable, so draw a mean and variance, take sqrt varianc,
    // and multiply
    r1_global ~ normal(0.0, 1.0);
    r2_global ~ inv_gamma(0.5, 0.5);

    r1_local ~ normal(0.0, 1.0);
    r2_local ~ inv_gamma(0.5, 0.5);

    return (r1_global * sqrt(r2_global)) * r1_local .* sqrt_vec(r2_local);
  }
}

data {
  int<lower=0> Nobs;  // number of data points
  int<lower=0> M;  // number of regressors
  int<lower=0> event[Nobs];  // 0 = no event, 1 = event
  vector[Nobs] time;  // time in trial
  matrix[Nobs, M] X;  // regressor matrix
  real<lower=0> dt;  // time bin size
}

transformed data {
  real<lower=0> tau_mu;
  real<lower=0> tau_al;

  tau_mu = 10.0;
  tau_al = 10.0;
}

parameters {
  // parameters used to make a half-Cauchy tau_s
  real<lower=0> tau_s1_raw;
  real<lower=0> tau_s2_raw;
  
  // parameters used to make local half-Cauchy tau_j's
  vector<lower=0>[M] tau1_raw;
  vector<lower=0>[M] tau2_raw;

  real alpha_raw;
  vector[M] beta_raw;

  real mu;  // beta0 parameter
}

transformed parameters {
  vector[M] beta;
  real alpha;
  vector[Nobs] talpha;

  beta = hs_prior_lp(tau_s1_raw, tau_s2_raw, tau1_raw, tau2_raw) .* beta_raw;
  alpha = exp(tau_al * alpha_raw);
  
  for (idx in 1:Nobs) {
    talpha[idx] = time[idx]^(alpha - 1);
  }
}

model {
  event ~ poisson(alpha * dt * talpha .* exp(mu + X * beta));

  beta_raw ~ normal(0.0, 1.0);
  alpha_raw ~ normal(0.0, 1.0);

  mu ~ normal(0.0, tau_mu);  
}
