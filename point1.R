
# Question 3 --------------------------------------------------------------

n = 250

#data model

p <- function(x, y){if(y==0){dunif(x, -3, 1) * 1/2}
    else if(y==1){dunif(x, -1, 3) * 1/2}
    else 0}

ys <- rbinom(n, 1, 0.5)
ys_test <- rbinom(n, 1, 0.5)
gen <- function(y){if(y==0){runif(1, -3, 1)}
    else if(y==1){runif(1, -1, 3)}}
gen <- Vectorize(gen)
xs <- gen(ys)
xs_test <- gen(ys_test)
par(mfrow=c(1,1))
plot.new()
hist(xs, freq = F, col = 'light blue', main = 'Marginal distribution of the variable X')
curve(p(x,0) + p(x, 1),col = 'orange', add = T, lwd = 3)
legend(1.2, 0.3, legend = c('Simulated Histogram', 'Theoretical PDF'), col = c('light blue', 'orange'), 
       lty = 1, lwd = 4, cex=0.8)

# Regression function
plot(xs, ys, col='red', pch=19)
reg.fun <- function(x){ p(x, 1)/(p(x,  0) + p(x, 1))}
curve(reg.fun, xlim = c(-5,5), lwd=2, col = 'green', add = T)

h.opt <- function(x) {if(reg.fun(x) > 0.5) {1}
                     else 0}
h.opt <- Vectorize(h.opt)
accuracy.Bayes = sum(h.opt(xs_test) == ys_test)/length(ys_test)
    

# Other Classifier --------------------------------------------------------

#val.indices = sample(seq(1, n), 50, replace=F)
train= data.frame(X = xs, y =  ys)
test= data.frame(X = xs_test, y =  ys_test)


model = glm(y ~ X, data = train, family = "binomial")
probabilities <- predict(model, newdata=test, type='response')
predicted.classes <- ifelse(probabilities > 0.5, 1, 0)
accuracy.Log = sum(predicted.classes == test$y)/length(test$y)


# Repeated Sampling -------------------------------------------------------

M = 1000
n = 250
total_acc_Bayes = c()
total_acc_Log = c()
for(i in 1:M){
    # generating samples of n=250 observations
    ys <- rbinom(n, 1, 0.5)
    ys_test <- rbinom(n, 1, 0.5)
    xs <- gen(ys)
    xs_test <- gen(ys_test)
    total_acc_Bayes = cbind(total_acc_Bayes,  c(sum(h.opt(xs_test) == ys_test)/length(ys_test)))
    # Logistic Regression
    train= data.frame(X = xs, y =  ys)
    test= data.frame(X = xs_test, y =  ys_test)
    model = glm(y ~ X, data = train, family = "binomial")
    probabilities <- predict(model, newdata=test, type='response')
    predicted.classes <- ifelse(probabilities > 0.5, 1, 0)
    total_acc_Log = cbind(total_acc_Log, c(sum(predicted.classes == test$y)/length(test$y)))
}
avg_acc_Bayes = mean(total_acc_Bayes)
avg_acc_Log = mean(total_acc_Log)
sd_acc_Bayes = sd(total_acc_Bayes)
sd_acc_Log = sd(total_acc_Log)
