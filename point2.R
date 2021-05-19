load("data/amazon_review_clean.RData")
library("MLmetrics")


differentiation = function(X.square, X.y, beta){
    return(X.square %*% beta - X.y)
}


X.square = t(X_tr) %*% X_tr
X.y = t(X_tr) %*% y_tr

beta = rnom(ncol(X_tr))


loss = MSE(y_pred = X_tr %*% beta, y_true = y_tr)
old.loss = loss + 1

while (old_loss - loss > 0.0000001) {
    old.loss = loss
    beta = beta - alpha * differentiation(square_matrix, lamble_matrix, beta)
    loss = MSE(y_pred = X_tr %*% beta, y_true = y_tr)
}