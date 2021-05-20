load("data/amazon_review_clean.RData")
library("MLmetrics")
library("Rfast")

set.seed(1234)

# changing labels of y vectors to book = TRUE, movie = FALSE
y_tr = (y_tr == "book")
y_te = (y_te == "book")

# differentiation variables:
gradient = function(n, X, X.y, X.t.beta){
    1/n * (t(X) %*% (1/(1+exp(-X.t.beta))) - X.y)
}

# X.square = mat.mult(t(X_tr), X_tr)

X.y = t(X_tr)%*%y_tr

# regression parameters
beta = rnorm(ncol(X_tr), sd=0.1)

# hyper parameters:
alpha = 0.1 # learning rate
tolerance = 0.00001 # tolerance for the convergence condition
n = nrow(X_tr)
X.t.beta = X_tr%*%beta
loss = 1/n * sum(log(1+exp(X.t.beta)) - y_tr*X.t.beta)
val.loss <- 1/n * sum(log(1+exp(X_te %*% beta)) - y_te*X_te %*% beta)
train.acc = Accuracy(y_pred = X.t.beta>0, y_true = y_tr)
test.acc = Accuracy(y_pred = X_te %*% beta>0, y_true = y_te)
train.accs = c(train.acc)
test.accs = c(test.acc)
val.hist <- c(val.loss)
loss.hist = c(loss)
it = 1
convergence = F
par(mfrow=c(1,2))
while (!convergence && it < 200) {
    # updating the parameters 
    beta = beta - alpha * gradient(n=nrow(X_tr), X_tr, X.y, X.t.beta)
    X.t.beta = X_tr%*%beta
    old.loss = loss
    loss = 1/n * sum(log(1+exp(X.t.beta)) - y_tr*X.t.beta)
    val.loss <- 1/n * sum(log(1+exp(X_te %*% beta)) - y_te*X_te %*% beta)
    # logging the loss history
    loss.hist <- cbind(loss.hist, c(loss))
    val.hist <- cbind(val.hist, c(val.loss))
    train.acc = Accuracy(y_pred = X.t.beta>0, y_true = y_tr)
    test.acc = Accuracy(y_pred = X_te %*% beta>0, y_true = y_te)
    train.accs <- cbind(train.accs, c(train.acc))
    test.accs <- cbind(test.accs, c(test.acc))
    # if change below the tolerance stop
    if (old.loss - loss < tolerance){convergence = T}
    plot(seq(length(loss.hist)), loss.hist, 
         type = "l", ylim = c(0,2), lwd = 3, col="dark red", main = "Loss during Gradient Descent",
         xlab = "epochs", ylab = "Cross-Entropy loss")
    lines(seq(length(loss.hist)), val.hist, type = "l", lwd = 3, col="dark blue")
    plot(seq(length(loss.hist)), train.accs, type = "l",
         ylim = c(0,1), lwd = 3, col="dark red", main = "Accuracy during Gradient Descent",
         xlab = "epochs", ylab = "Accuracy")
    lines(seq(length(loss.hist)), test.accs, type = "l", lwd = 3, col="dark blue")
    it = it + 1
}
# plotting the results
plot(seq(length(loss.hist)), loss.hist, 
     type = "l", ylim = c(0,2), lwd = 3, col="dark red", main = "Loss during Gradient Descent",
     xlab = "epochs", ylab = "Cross-Entropy loss")
# metrics on train and test
train.loss = loss.hist[length(loss.hist)]
test.loss = 1/n * sum(log(1+exp(X_te %*% beta)) - y_te*X_te %*% beta)
train.acc = Accuracy(y_pred = X.t.beta>0, y_true = y_tr)
test.acc = Accuracy(y_pred = X_te %*% beta>0, y_true = y_te)
