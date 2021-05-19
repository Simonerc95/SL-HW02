load("data/amazon_review_clean.RData")
library("MLmetrics")

# changing labels of y vectors to book = TRUE, movie = FALSE
y_tr = (y_tr == "book")
y_te = (y_te == "book")
differentiation = function(n, X.square, X.y, beta){
    2/n * (X.square %*% beta - X.y)
}


X.square = t(X_tr) %*% X_tr
X.y = t(X_tr) %*% y_tr

beta = rnorm(ncol(X_tr))

# learning rate
alpha = 0.001

loss = MSE(y_pred = X_tr %*% beta, y_true = y_tr)
convergence = F
loss.hist = c(loss)
while (!convergence) {
    beta = beta - alpha * differentiation(n=nrow(X_tr), X.square, X.y, beta)
    old.loss = loss
    loss = MSE(y_pred = X_tr %*% beta, y_true = y_tr)
    loss.hist <- cbind(loss.hist, c(loss))
    if (old.loss - loss < 0.01){convergence = T}
}

plot(seq(40), loss.hist[390:429])
