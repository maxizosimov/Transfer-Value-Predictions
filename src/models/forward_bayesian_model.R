# This contains code to fit and evaluate a Bayesian model on midfielders.
source('src/models/bayesian_utils.R')

library(MLmetrics) # Contains R^2 function
set.seed(42) # for reproducibility

# Read in real player stats and values
position <- 'F'

res <- load_position_real_data(position)
train_data <- res$train_data
test_data <- res$test_data

# One-hot encode league
# Get league levels
league_levels <- c("Bundesliga", "Serie_A", "Ligue_1", "La_Liga", "EPL")

train_data <- one_hot_encode_league(train_data, league_levels)
test_data <- one_hot_encode_league(test_data, league_levels)

# Winsorize
winsor_params <- read.csv(sprintf('src/data/%s_winsorize_params.csv', position))
res <- winsorize(train_data, test_data, winsor_params)
train_data <- res$train_data
test_data <- res$test_data

# Get train/test matrices

# Use these rather than indexing
continuous_cols <- c('xG_per_90',
                     'xA_per_90',
                     'xGChain_per_90',
                     'age',
                     'year')

drop_cols <- c("player_id", "player_name", "date", "value")

# Store min/max from training
col_mins <- apply(train_data[, continuous_cols], 2, min)
col_maxs <- apply(train_data[, continuous_cols], 2, max)

res <- get_X_y(train_data, continuous_cols, drop_cols, col_mins, col_maxs)
X_train <- res$X
y_train <- res$y

res <- get_X_y(test_data, continuous_cols, drop_cols, col_mins, col_maxs)
X_test <- res$X
y_test <- res$y

# Using OLS
w_hat <- solve(t(X_train)%*%X_train)%*%t(X_train)%*%y_train
w_hat

# Normal model with unknown (common) variance
# Fit via Gibbs sampling

# Define priors

# To guide choice of intercept, which when everything else is 0 should reflect
# value for a young player from an early year with poor performance
(train_data[train_data$age < 21
            & train_data$year < 2015,]) # Looks like about 1 million

prior_w <- matrix(c(0.5, # xg - larger since forwards should score more goals
                    0.5, # xa - larger since forwards should get more assists
                    0.7, # xgchain -
                    0.3, # year - small positive for inflation
                    -3, # age - strong negative
                    0, # Serie A - equivalent to Bundesliga
                    -0.2, # Ligue 1 - worse than Bunesliga
                    1, # PL - most expensive league
                    .7, # La liga - next most after PL
                    log(1000000) # intercept - when everything else is 0 (min stats + Bundesliga)
), ncol = 1)

p <- nrow(prior_w)

prior_Sigma <- diag(p)*1

# Now need priors for sigma_y
prior_alpha <- 1
prior_beta <- 1

# Run gibbs
no_samples <- 1000
res <- gibbs(X_train, y_train, prior_w, prior_Sigma, prior_alpha, prior_beta, no_samples=no_samples, burn=.18)
gibbs_samples_w <- res$w
gibbs_samples_sigmay <- res$sigmay

# check convergence and mixing of weights
head(gibbs_samples_w)
convergence_acf_plots(gibbs_samples_w)

# Get test predictions
y_test_pred <- get_preds(X_test)

# Summary metrics
rmse(exp(y_test), exp(y_test_pred))
# Calculate R^2
R2_Score(y_pred = y_test_pred, y_true = y_test)

# Plot predictions vs actuals
plot_preds_vs_actuals(y_test_pred, y_test,ylim=c(0,2e8))

test_data[exp(y_test_pred) > 100000000, ] # Who is predicted over 100mil?

# Look at posterior for one player
player_name <- "Lamine Yamal"
latest_player_idx <- tail(which(test_data$player_name == player_name),1)

player_preds <- player_posterior(X_test[latest_player_idx,], gibbs_samples_w, gibbs_samples_sigmay)

plot_player_posterior(X_test[latest_player_idx,], player_name, exp(y_test[latest_player_idx,]),
                      gibbs_samples_w, gibbs_samples_sigmay, breaks=20)

plot_player_value_over_time(player_name, X_test, test_data, gibbs_samples_w, gibbs_samples_sigmay, ci=0.9)

# Credible intervals
quantile(player_preds, c(0.1, 0.9))  # 90% credible interval
exp(quantile(player_preds, c(0.1, 0.9)))  # back to euros

############################################### What if given PREDICTED performance from LSTM?
differenced <- 'True'

full_test_data <- read.csv(sprintf("src/data/%s_differenced_set_%s_predictions_real_values.csv", position, differenced))
head(full_test_data)

# One-hot encode league
full_test_data <- one_hot_encode_league(full_test_data, league_levels)

# Get matrices
res <- get_X_y(full_test_data, continuous_cols, drop_cols, col_mins, col_maxs)
X_full_test <- res$X
y_full_test <- res$y

# Make predictions
y_full_test_pred <- get_preds(X_full_test)

# Summary metrics
rmse(exp(y_full_test), exp(y_full_test_pred))
# Calculate R^2
R2_Score(y_pred = y_full_test_pred, y_true = y_full_test)

# Plot predictions vs actuals
plot_preds_vs_actuals(y_full_test_pred, y_full_test,ylim=c(0,5e7))

full_test_data[exp(y_full_test_pred) > 100000000, ] # Who is predicted over 100mil?

# Look at posterior for one player
player_name <- "Lamine Yamal"
latest_player_idx <- tail(which(full_test_data$player_name == player_name),1)

player_preds <- player_posterior(X_full_test[latest_player_idx,], gibbs_samples_w, gibbs_samples_sigmay)

plot_player_posterior(X_full_test[latest_player_idx,], player_name, exp(y_full_test[latest_player_idx,]),
                      gibbs_samples_w, gibbs_samples_sigmay, breaks=20)

plot_player_value_over_time(player_name, X_full_test, full_test_data, gibbs_samples_w, gibbs_samples_sigmay, ci=0.9)

# Credible intervals
quantile(player_preds, c(0.1, 0.9))  # 90% credible interval
exp(quantile(player_preds, c(0.1, 0.9)))  # back to euros