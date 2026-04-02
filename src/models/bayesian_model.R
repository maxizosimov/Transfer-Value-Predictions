library(scales)  # For Min-Max scaling
library(mvnfast) # For MVN calcs
library(MCMCpack) # For inverse gamma sampling
library(MLmetrics) # Contains R^2 function
library(data.table) # For single feature one-hot encoding
# Train the model: read in real player stats and values

# Get real data
real_stats_values <- read.csv("src/data/F_stats_values.csv")
head(real_stats_values)

X <- data.matrix(real_stats_values[,-c(1,2,3,6,14)])
y <- data.matrix(real_stats_values[,14])

# Log-transform y
y <- log(y)

# Scale X using min/max scaling
X <- apply(X, MARGIN = 2, function(x) rescale(x, to = c(0, 1)))

# Add bias column to X
X <- cbind(X, bias = 1)

# Train/test split
train_size = 0.8
train_indices <- sample(1:nrow(X), size = round(train_size * nrow(X)), replace = F)

X_train <- X[train_indices, ]
y_train <- y[train_indices]
X_test <- X[-train_indices, ]
y_test <- y[-train_indices]

# Using OLS
w_hat <- solve(t(X_train)%*%X_train)%*%t(X_train)%*%y_train
w_hat

# Look at a few predictions
y_test_pred <- X_test %*% w_hat
head(y_test_pred)
head(real_stats_values[-train_indices,])

# Calculate MSE
mse <- function(actuals, preds) {
    mean((preds - actuals)**2)
}

mse(y_test, y_test_pred)
# Calculate R^2
R2_Score(y_pred = y_test_pred, y_true = y_test)

# Normal model with unknown (common) variance
# Fit via Gibbs sampling


# Define priors
prior_w <- matrix(c(1, # All should positively contribute
                    1,
                    1,
                    1, 
                    1,
                    1,
                    1,
                    0 # intercept of 0
                    ), ncol = 1)

prior_Sigma <- diag(8)*100 # uncertain though; use big sd = 10

# Now need priors for sigma_y
prior_alpha <- 1
prior_beta <- 1
# will need the sample size
n <- length(y_train)

# Set up markov chain matrices
no_samples <- 10000
gibbs_samples_w <- matrix(0, nrow = no_samples, ncol = 8)
gibbs_samples_sigmay <- matrix(0, nrow = no_samples, ncol = 1)

#initialize
w_tilde <- matrix(0, nrow = 8, ncol = 1)
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
plot(gibbs_samples_w[,1], type = "l") # change the index to examine the 8 weights
acf(gibbs_samples_w[,1]) # change the index to examine the 8 weights

# check convergence and mixing of variance term
head(gibbs_samples_sigmay)
plot(gibbs_samples_sigmay, type = "l")
acf(gibbs_samples_sigmay)

# visualize posteriors?
hist(gibbs_samples_sigmay, main = "Posterior of Variance of Market Value", xlim=c(min(gibbs_samples_sigmay), max(gibbs_samples_sigmay)))
hist(gibbs_samples_w[,1], main = "Posterior of goals_per_90 Slope", xlim=c(min(gibbs_samples_w[,1]), max(gibbs_samples_w[,1])))
hist(gibbs_samples_w[,2], main = "Posterior of xG_per_90 Slope", xlim=c(min(gibbs_samples_w[,2]), max(gibbs_samples_w[,2])))
hist(gibbs_samples_w[,3], main = "Posterior of assists_per_90 Slope", xlim=c(min(gibbs_samples_w[,3]), max(gibbs_samples_w[,3])))
hist(gibbs_samples_w[,4], main = "Posterior of xA_per_90 Slope", xlim=c(min(gibbs_samples_w[,4]), max(gibbs_samples_w[,4])))
hist(gibbs_samples_w[,5], main = "Posterior of key_passes_per_90 Defense Slope", xlim=c(min(gibbs_samples_w[,5]), max(gibbs_samples_w[,5])))
hist(gibbs_samples_w[,6], main = "Posterior of xGChain_per_90 Slope", xlim=c(min(gibbs_samples_w[,6]), max(gibbs_samples_w[,6])))
hist(gibbs_samples_w[,7], main = "Posterior of xGBuildup_per_90 Slope", xlim=c(min(gibbs_samples_w[,7]), max(gibbs_samples_w[,7])))
hist(gibbs_samples_w[,7], main = "Posterior of Intercept", xlim=c(min(gibbs_samples_w[,8]), max(gibbs_samples_w[,8])))

# Generating predictive distribution for Dembele
demb_hat <- c()
for(i in 1:nrow(gibbs_samples_w)){
    demb_hat <- c(demb_hat, rnorm(n = 1, mean = t(gibbs_samples_w[i,])%*%X[1,], sd = sqrt(gibbs_samples_sigmay[i,])))
}
hist(demb_hat, main = "Post Pred Dist of Dembele (2021)")
mean(demb_hat)
y[1]