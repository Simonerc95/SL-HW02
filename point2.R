load("data/amazon_review_clean.RData")
library("MLmetrics")
library("Rfast")

set.seed(1234)

# changing labels of y vectors to book = TRUE, movie = FALSE
y_tr = (y_tr == "book")
y_te = (y_te == "book")

# differentiation variables:
differentiation = function(n, X.square, X.y, beta){
    2/n * (mat.mult(X.square, matrix(beta)) - X.y)
}

X.square = mat.mult(t(X_tr), X_tr)
X.y = mat.mult(t(X_tr), matrix(y_tr))

# regression parameters
beta = rnorm(ncol(X_tr))

# hyper parameters:
alpha = 0.001 # learning rate
tolerance = 0.1 # tolerance for the convergence condition

loss = MSE(y_pred = mat.mult(X_tr, matrix(beta)), y_true = y_tr)
convergence = F
loss.hist = c(loss)
while (!convergence) {
    # updating the parameters 
    beta = beta - alpha * differentiation(n=nrow(X_tr), X.square, X.y, beta)
    
    old.loss = loss
    loss = MSE(y_pred = mat.mult(X_tr, matrix(beta)), y_true = y_tr)
    
    # logging the loss history
    loss.hist <- cbind(loss.hist, c(loss))
    
    # if change below the tolerance stop
    if (old.loss - loss < tolerance){convergence = T} 
}

# plotting the results
plot(seq(length(loss.hist)), loss.hist)


train.loss = loss.hist[length(loss.hist)]
test.loss = MSE(y_pred = mat.mult(X_te, matrix(beta)), y_true = y_te)
train.acc = Accuracy(y_pred = mat.mult(X_tr, matrix(beta))>.5, y_true = y_tr)
test.acc = Accuracy(y_pred = mat.mult(X_te, matrix(beta))>.5, y_true = y_te)
