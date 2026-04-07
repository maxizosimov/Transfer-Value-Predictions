library(scales)  # For Min-Max scaling
library(mvnfast) # For MVN calcs
library(MCMCpack) # For inverse gamma sampling
library(MLmetrics) # Contains R^2 function
library(data.table) # For single feature one-hot encoding
set.seed(42)

# Train the model: read in real player stats and values

position <- 'M'

# Get real data
train_data <- read.csv(sprintf("src/data/%s_stats_values_train.csv", position))
test_data <- read.csv(sprintf("src/data/%s_stats_values_test.csv", position))
head(train_data)

# One-hot encode league
# Get league levels from training data only
league_levels <- unique(train_data$league)

# Force test data to use same levels
train_data$league <- factor(train_data$league, levels = league_levels)
test_data$league <- factor(test_data$league, levels = league_levels)

# One-hot encode both using same reference level
train_league_cols <- model.matrix(~ league - 1, data = train_data)[, -1]
test_league_cols <- model.matrix(~ league - 1, data = test_data)[, -1]

# Combine
train_data <- cbind(train_data[, -4], train_league_cols)
test_data <- cbind(test_data[, -4], test_league_cols)

# Get train/test matrices
X_train <- data.matrix(train_data[,-c(1,2,3,11)])
y_train <- data.matrix(train_data[,11])

X_test <- data.matrix(test_data[,-c(1,2,3,11)])
y_test <- data.matrix(test_data[,11])

# Log-transform y
y_train <- log(y_train)
y_test <- log(y_test)

# Winsorize
winsor_params <- read.csv(sprintf('src/data/%s_winsorize_params.csv', position))
caps <- setNames(winsor_params$cap, winsor_params$stat)

for (i in 1:7) {
    X_train[, i] <- pmin(X_train[, i], caps[i])
    X_test[, i] <- pmin(X_test[, i], caps[i])
}

# Scale X using min/max scaling

continuous_cols <- c('goals_per_90', 'xG_per_90', 'assists_per_90', 
                     'xA_per_90', 'key_passes_per_90', 'xGChain_per_90',
                     'xGBuildup_per_90', 'age', 'year')

# Store min/max from training
col_mins <- apply(X_train[, continuous_cols], 2, min)
col_maxs <- apply(X_train[, continuous_cols], 2, max)

for (col in continuous_cols) {
    X_train[, col] <- (X_train[, col] - col_mins[col]) / (col_maxs[col] - col_mins[col])
    X_test[, col] <- (X_test[, col] - col_mins[col]) / (col_maxs[col] - col_mins[col])
}


X_train <- apply(X_train, MARGIN = 2, function(x) rescale(x, to = c(0, 1)))
X_test <- apply(X_test, MARGIN = 2, function(x) rescale(x, to = c(0, 1)))

# Add bias column to X
X_train <- cbind(X_train, bias = 1)
X_test <- cbind(X_test, bias = 1)

# Using OLS
w_hat <- solve(t(X_train)%*%X_train)%*%t(X_train)%*%y_train
w_hat

# Normal model with unknown (common) variance
# Fit via Gibbs sampling

# Define priors

mean(y_train)

prior_w <- matrix(c(1, # goals
                    1, # xg
                    1, # assists
                    1, # xa
                    1, # key passes
                    1, # xgchain
                    1, # xgbuildup
                    1, # year
                    -1, # age
                    0, # Ligue 1
                    0, # Bundesliga
                    1, # PL
                    0, # La liga
                    mean(y_train) # use mean of training data for intercept
                    ), ncol = 1)

p <- nrow(prior_w)

prior_Sigma <- diag(p)*3 # Tighter variances for regularization 

# Now need priors for sigma_y
prior_alpha <- 1
prior_beta <- 1
# will need the sample size
n <- length(y_train)

# Set up markov chain matrices
no_samples <- 1000
gibbs_samples_w <- matrix(0, nrow = no_samples, ncol = p)
gibbs_samples_sigmay <- matrix(0, nrow = no_samples, ncol = 1)

#initialize
w_tilde <- matrix(0, nrow = p, ncol = 1)
sigma2_tilde <- 1

# run the sampler
for(i in 1:no_samples){
    # conditional posterior parameters for w
    V = solve((1/sigma2_tilde)*t(X_train)%*%X_train + solve(prior_Sigma))
    M = V%*%((1/sigma2_tilde)*t(X_train)%*%y_train + solve(prior_Sigma)%*%prior_w)
    # sample weights
    w_tilde <- t(rmvn(1, M, V)) # rmvn gives w^T
    
    # conditional posterior parameters for sigma2
    A = (2*prior_alpha + n)/2
    B = (2*prior_beta + t(y_train - X_train%*%w_tilde)%*%(y_train - X_train%*%w_tilde))/2
    # sample sigma2
    sigma2_tilde <- rinvgamma(1, A, B)
    
    # store them
    gibbs_samples_w[i,] <- w_tilde
    gibbs_samples_sigmay[i,] <- sigma2_tilde
}

# check convergence and mixing of weights
head(gibbs_samples_w)
plot(gibbs_samples_w[,1], type = "l") # change the index to examine the weights
acf(gibbs_samples_w[,1]) # change the index to examine the weights

# check convergence and mixing of variance term
head(gibbs_samples_sigmay)
plot(gibbs_samples_sigmay, type = "l")
acf(gibbs_samples_sigmay)

# visualize posteriors?
hist(gibbs_samples_sigmay, main = "Posterior of Variance of Market Value", xlim=c(min(gibbs_samples_sigmay), max(gibbs_samples_sigmay)))
hist(gibbs_samples_w[,1], main = "Posterior of _ Slope", xlim=c(min(gibbs_samples_w[,1]), max(gibbs_samples_w[,1])))

# Get test predictions
get_preds <- function(X) {
    test_preds <- matrix(0, nrow = nrow(X), ncol = no_samples)
    for(i in 1:no_samples) {
        test_preds[,i] <- rmvn(n=1, mu=X%*%gibbs_samples_w[i,], sigma=diag(nrow(X))*gibbs_samples_sigmay[i])
    }
    rowMeans(test_preds)
}

y_test_pred <- get_preds(X_test)

rmse <- function(actuals, preds) {
    sqrt(mean((actuals - preds)^2))
}

rmse(exp(y_test), exp(y_test_pred))
# Calculate R^2
R2_Score(y_pred = y_test_pred, y_true = y_test)

# Plot predictions vs actuals
plot(y_test, y_test_pred)
abline(a = 0, b = 1, col = 'red', lwd = 2)

sum(exp(y_test_pred) > 100000000)  # predictions over 100m

tail(test_data[exp(y_test_pred) > 100000000, ]) # Who are the outliers

# Look at posterior for one player
player_preds <- c()
for(i in 1:no_samples){
    player_preds <- c(player_preds, rnorm(n = 1, mean = t(gibbs_samples_w[i,])%*%X_test[3,], sd = sqrt(gibbs_samples_sigmay[i])))
}
hist(player_preds, main = "Post Pred Dist of Player")

# Credible intervals
quantile(player_preds, c(0.05, 0.95))  # 90% credible interval
exp(quantile(player_preds, c(0.05, 0.95)))  # back to euros

############################################### What if given PREDICTED performance from LSTM?
differenced <- 'True'

full_test_data <- read.csv(sprintf("src/data/%s_differenced_set_%s_predictions_real_values.csv", position, differenced))
head(full_test_data)

full_test_data$league <- factor(full_test_data$league, levels = league_levels)

# One-hot encode using same reference level
ft_league_cols <- model.matrix(~ league - 1, data = full_test_data)[, -1]

# Combine
full_test_data <- cbind(full_test_data[, -4], ft_league_cols)

# Get matrices
X_full_test <- data.matrix(full_test_data[,-c(1,2,3,11)])
y_full_test <- data.matrix(full_test_data[,11])

# Log-transform y
y_full_test <- log(y_full_test)

# Scale X using min/max scaling
for (col in continuous_cols) {
    X_full_test[, col] <- (X_full_test[, col] - col_mins[col]) / (col_maxs[col] - col_mins[col])
}

# Add bias column to X
X_full_test <- cbind(X_full_test, bias = 1)


# Make predictions
y_full_test_pred <- get_preds(X_full_test)

rmse(exp(y_full_test), exp(y_full_test_pred))
# Calculate R^2
R2_Score(y_pred = y_full_test_pred, y_true = y_full_test)

# Plot predictions vs actuals
plot(y_full_test, y_full_test_pred)
abline(a = 0, b = 1, col = 'red', lwd = 2)

sum(exp(y_full_test_pred) > 100000000)  # predictions over 100m

# Look at posterior for one player
player_preds <- c()
for(i in 1:no_samples){
    player_preds <- c(player_preds, (rnorm(n = 1, mean = t(gibbs_samples_w[i,])%*%X_full_test[1,], sd = sqrt(gibbs_samples_sigmay[i]))))
}
hist(player_preds, main = "Post Pred Dist of Player")

# Credible intervals
quantile(player_preds, c(0.05, 0.95))  # 90% credible interval
exp(quantile(player_preds, c(0.05, 0.95)))  # back to euros