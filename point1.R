
# Question 3 --------------------------------------------------------------

n = 250

#data model

p <- function(x, y){if(y==0){dunif(x, -3, 1) * 1/2}
    else if(y==1){dunif(x, -1, 3) * 1/2}
    else 0}
curve(p(x,0) + p(x, 1), xlim = c(-5, 5),col = 'red')

ys <- rbinom(n, 1, 0.5)
gen <- function(y){if(y==0){runif(1, -3, 1)}
    else if(y==1){runif(1, -1, 3)}}
gen <- Vectorize(gen)
xs <- gen(ys)
plot.new()
hist(xs, freq = F, col = 'light blue', main = 'Marginal distribution of the variable X')
curve(p(x,0) + p(x, 1),col = 'orange', add = T, lwd = 3)
legend(1.2, 0.3, legend = c('Simulated EPDF', 'Theoretical PDF'), col = c('light blue', 'orange'), 
       lty = 1, lwd = 4, cex=0.8)

# Regression function
plot(xs, ys, col='red', pch=19)
reg.fun <- function(x){ p(x, 1)/(p(x,  0) + p(x, 1))}
curve(reg.fun, xlim = c(-5,5), lwd=2, col = 'green', add = T)

h.opt <- function(x) {if(reg.fun(x) > 0.5) {1}
                     else 0}
h.opt <- Vectorize(h.opt)
accuracy.Bayes = sum(h.opt(xs) == ys)/length(ys)

# Other classifier
val.indices = sample(seq(1, n), 50, replace=F)
train= data.frame(X = xs[-val.indices], y =  ys[-val.indices])
val= data.frame(X = xs[val.indices], y =  ys[val.indices])


model = glm(y ~ X, data = train, family = "binomial")
probabilities <- predict(model, newdata=val, type='response')
predicted.classes <- ifelse(probabilities > 0.5, 1, 0)
accuracy.Log = sum(predicted.classes == val$y)/length(val$y)
