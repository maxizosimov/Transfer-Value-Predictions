# This contains helpers for use in training a Bayesian either on midfielders or forwards.

library(scales)  # For Min-Max scaling
library(mvnfast) # For MVN calcs
library(MCMCpack) # For inverse gamma sampling
library(data.table) # For single feature one-hot encoding
library(ggplot2) # For plots

load_position_real_data <- function(position) {
    # This produces the train/test dataframes for real data at the given position
    train_data <- read.csv(sprintf("src/data/%s_stats_values_train.csv", position))
    test_data <- read.csv(sprintf("src/data/%s_stats_values_test.csv", position))
    list(train_data=train_data, test_data=test_data)
}

one_hot_encode_league <- function(data, league_levels) {
    # This one-hot encodes the league using the given league_levels for consistency
    
    # Force data to use same levels
    data$league <- factor(data$league, levels = league_levels)
    
    league_cols <- model.matrix(~ league - 1, data = data)[, -1]
    
    # Drop the original league column and add one-hot encoded version
    data <- cbind(data[, names(data) != "league"], league_cols)
    
    data
}

winsorize <- function(train_data, test_data, winsor_params) {
    # This winsorizes the given train and test dfs
    caps <- setNames(winsor_params$cap, winsor_params$stat)
    stat_cols <- names(caps)
    
    for (col in stat_cols) {
        train_data[, col] <- pmin(train_data[, col], caps[col])
        test_data[, col]  <- pmin(test_data[, col],  caps[col])
    }
    
    list(train_data=train_data, test_data=test_data)
}

get_X_y <- function(data, continuous_cols, drop_cols, col_mins, col_maxs) {
    # This helper produces the X and y matrices corresponding to the given data
    # It min/max scales X according to the given col min/max values, adds
    # a bias, and log transforms y.
    
    for (col in continuous_cols) {
        data[, col] <- (data[, col] - col_mins[col]) / (col_maxs[col] - col_mins[col])
    }
    
    X <- data.matrix(data[, !names(data) %in% drop_cols])
    y <- data.matrix(data[, "value"])
    
    # Log-transform y
    y <- log(y)
    
    # Add bias
    X <- cbind(X, bias = 1)
    
    list(X=X, y=y)
}

gibbs <- function(X_train, y_train, prior_w, prior_Sigma, prior_alpha, prior_beta, no_samples = 1000, burn=.18) {
    # Runs Gibbs sampling on the given X/y pair, and with the given prior choices
    # for the weights and uncertainties, as well as for the covariance of y.
    # Returns the sampled ws and Sigmas.
    p <- nrow(prior_w)
    n <- nrow(y_train)
    
    total_samples = floor(no_samples / (1 - burn))
    m = floor(total_samples * burn) # Discard amount
    
    # Set up markov chain matrices
    gibbs_samples_w <- matrix(0, nrow = total_samples, ncol = p)
    gibbs_samples_sigmay <- matrix(0, nrow = total_samples, ncol = 1)
    
    #initialize
    w_tilde <- matrix(0, nrow = p, ncol = 1)
    sigma2_tilde <- 1
    
    # run the sampler
    for (i in 1:total_samples) {
        # conditional posterior parameters for w
        V <- solve((1 / sigma2_tilde) * t(X_train) %*% X_train + solve(prior_Sigma))
        M <- V %*% ((1 / sigma2_tilde) * t(X_train) %*% y_train + solve(prior_Sigma) %*% prior_w)
        # sample weights
        w_tilde <- t(rmvn(1, M, V))
        
        # conditional posterior parameters for sigma2
        A <- (2 * prior_alpha + n) / 2
        B <- (2 * prior_beta + t(y_train - X_train %*% w_tilde) %*% (y_train - X_train %*% w_tilde)) / 2
        sigma2_tilde <- rinvgamma(1, A, B)
        
        # store them
        gibbs_samples_w[i, ] <- w_tilde
        gibbs_samples_sigmay[i, ] <- sigma2_tilde
    }
    
    list(w= gibbs_samples_w[(m+1):total_samples,], sigmay = gibbs_samples_sigmay[(m+1):total_samples,])
}

get_preds <- function(X) {
    # Get test predictions
    test_preds <- matrix(0, nrow = nrow(X), ncol = no_samples)
    for(i in 1:no_samples) {
        test_preds[,i] <- rmvn(n=1, mu=X%*%gibbs_samples_w[i,], sigma=diag(nrow(X))*gibbs_samples_sigmay[i])
    }
    rowMeans(test_preds)
}

rmse <- function(actuals, preds) {
    # Root mean squared error
    sqrt(mean((actuals - preds)^2))
}

player_posterior <- function(X_row, gibbs_samples_w, gibbs_samples_sigmay) {
    # Produces sampled log-value predictions for a given row.
    
    player_preds <- c()
    
    no_samples <- nrow(gibbs_samples_w)
    for(i in 1:no_samples){
        player_preds <- c(player_preds, rnorm(n = 1, mean = t(gibbs_samples_w[i,])%*%X_row, sd = sqrt(gibbs_samples_sigmay[i])))
    }
    
    player_preds
}

convergence_acf_plots <- function(samples) {
    # Convergence and ACF plots per feature
    for (i in 1:ncol(samples)) {
        plot(samples[,i], type = "l", main=sprintf("Convergence of feature %d",i))
        acf(samples[,i], main=sprintf("ACF of %d",i))
    }
}

# Plot predictions vs actuals
plot_preds_vs_actuals <- function(predictions,actuals,ylim) {
    # To euros
    pred_euros <- exp(predictions)
    actual_euros <- exp(actuals)
    
    plot(actual_euros, pred_euros, xlab  = "Actual Value (euros)", ylab  = "Predicted Value (euros)", col="red",ylim=ylim)
    
    # Perfect prediction line
    abline(a = 0, b = 1, col = "blue", lwd = 2)
}

plot_player_posterior <- function(player_row, player_name, actual_value,
                                  gibbs_samples_w, gibbs_samples_sigmay, breaks) {
    # Get posterior samples
    preds <- player_posterior(player_row, gibbs_samples_w, gibbs_samples_sigmay)
    pred_euros <- exp(preds)
    
    # Use ggplot
    df <- data.frame(pred_euros = pred_euros)
    
    ggplot(df, aes(x = pred_euros)) +
        geom_histogram(bins = breaks, color="black") +
        geom_vline(aes(xintercept = actual_value,
                   colour = "Actual Value")) +
        scale_colour_manual(name = "Legend", values = c("Actual Value" = "red")) +
        scale_x_continuous(labels = function(x) formatC(x, format = "f", digits = 0, big.mark = ",")) +
        labs(title = sprintf("Posterior Predictive Distribution: %s", player_name),
             x = "Predicted Market Value (euros)")
}

plot_player_value_over_time <- function(player_name, X, data,
                                        gibbs_samples_w, gibbs_samples_sigmay,
                                        ci = 0.9) {
    # Plots a player's predicted value over time with the credible interval bounds.
    
    # Get indices for this player
    player_mask <- data$player_name == player_name
    player_X <- X[player_mask, ]
    player_dates <- as.Date(data$date[player_mask])
    
    upper_b <- ci + (1 - ci) / 2
    lower_b <- (1 - ci) / 2
    
    # Get posterior samples for each time point
    n_points <- sum(player_mask)
    
    means  <- numeric(n_points)
    lowers <- numeric(n_points)
    uppers <- numeric(n_points)
    
    for (i in 1:n_points) {
        preds <- exp(player_posterior(player_X[i, ], gibbs_samples_w, gibbs_samples_sigmay))
        means[i] <- mean(preds)
        lowers[i] <- quantile(preds, lower_b)
        uppers[i] <- quantile(preds, upper_b)
    }
    
    # Make plot
    df <- data.frame(
        date = player_dates,
        mean = means,
        lower = lowers,
        upper = uppers
    )
    
    ggplot(df, aes(x = date)) +
        geom_ribbon(aes(ymin = lower, ymax = upper), 
                    fill="lightblue") +
        geom_line(aes(y = mean), color = "blue", linewidth = 1) +
        scale_y_continuous(labels = function(x) formatC(x, format = "f", digits = 0, big.mark = ",")) +
        labs(x= "Date",
             y = "Predicted Market Value (euros)")
}