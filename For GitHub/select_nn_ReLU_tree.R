#### MIT 6.867 Breast Cancer Subtyping
### Select NN intermediate features after ReLU
# Xihao Li

bcr_train_relu <- read.table("~/Dropbox/6.867project/data/nn_intermediate_relu_train.txt",header=FALSE)
dim(bcr_train_relu)
# [1]  329 1001

bcr_test_relu <- read.table("~/Dropbox/6.867project/data/nn_intermediate_relu_test.txt",header=FALSE)
dim(bcr_test_relu)
# [1]  164 1001

# Count the number of training samples that have non-zero entry for each feature
colsum_train <- apply(bcr_train_relu, 2, function(x){sum(x!=0)})
hist(colsum_train,breaks=20)
summary(colsum_train)
table(colsum_train)

sum(colsum_train >= 165) # [1] 54 (select features with at least a half obs non-zero)
colsum_train >= 165

bcr_train_relu_selected <- bcr_train_relu[,colsum_train >= 165]
head(bcr_train_relu_selected)
dim(bcr_train_relu_selected) # [1] 329  54
write.table(bcr_train_relu_selected,"~/Dropbox/6.867project/data/nn_intermediate_relu_train_selected.txt")

bcr_test_relu_selected <- bcr_test_relu[,colsum_train >= 165]
head(bcr_test_relu_selected)
dim(bcr_test_relu_selected) # [1] 164  54
write.table(bcr_test_relu_selected,"~/Dropbox/6.867project/data/nn_intermediate_relu_test_selected.txt")

###################################
#### Tree-based methods
### Final model: based on at least 100 obs. not equal to 0, using NN imtermediate step after ReLU (1000 hidden units)
###################################

setwd("~/Dropbox/6.867project/Milestone_6/figures/")
###### Read the train set from 1000 NN selected features
bcr_train_nn_relu <- read.table("~/Dropbox/6.867project/data/nn_intermediate_relu_train_selected.txt",header=TRUE)
dim(bcr_train_nn_relu) # [1] 329 54
bcr_train_nn_relu[1:5,1:10]
bcr_train_nn_relu$V1001 <- as.factor(bcr_train_nn_relu$V1001)
colnames(bcr_train_nn_relu)[54] <- "y"
dim(bcr_train_nn_relu) # [1] 329 54
sum(bcr_train_nn_relu$y == 1) # 169
sum(bcr_train_nn_relu$y == 0) # 160
dim(bcr_train_nn_relu)

###### Read the test set
bcr_test_nn_relu <- read.table("~/Dropbox/6.867project/data/nn_intermediate_relu_test_selected.txt",header=TRUE)
dim(bcr_test_nn_relu) # [1] 164 54
bcr_test_nn_relu[1:5,1:10]
bcr_test_nn_relu$V1001 <- as.factor(bcr_test_nn_relu$V1001)
colnames(bcr_test_nn_relu)[54] <- "y"
dim(bcr_test_nn_relu) # [1] 164 54
sum(bcr_test_nn_relu$y == 1) # 77
sum(bcr_test_nn_relu$y == 0) # 87
dim(bcr_test_nn_relu)

###############
# Tree based methods
###############
library(tree)
tree.single <- tree(y ~ ., data = bcr_train_nn_relu)
summary(tree.single)

pdf("Tree_single_nn_selected_relu.pdf",width = 10, height = 6)
plot(tree.single, main = "Single Decision Tree (Using ReLU selected NN features)")
text(tree.single, pretty = 0)
dev.off()

# Print a detailed summary of the tree
tree.single

tree.pred = predict(tree.single, newdata = bcr_test_nn_relu, type = "class") # Note: there is randomness here
(test=with(bcr_test_nn_relu, table(tree.pred, y)))

(test[2,2] + test[1,1])/sum(test) # [1] 0.7560976

# Pruning
set.seed(1)
bcr.tree.cv = cv.tree(tree.single, FUN = prune.misclass) # use number of misclassifications as the objective function
bcr.tree.cv # size is the total number of leaf nodes, k is the cost-complexity parameter


min_idx = which.min(bcr.tree.cv$dev) # The first index with smallest dev
bcr.tree.cv$size[min_idx]
prune.train = prune.misclass(tree.single, best = bcr.tree.cv$size[min_idx])

pdf("Tree_prune_nn_selected_relu.pdf",width=10,height=6)
plot(prune.train, main = "Pruned Decision Tree (Using ReLU selected NN features)")
text(prune.train, pretty = 0)
dev.off()

tree.prune.pred = predict(prune.train, newdata = bcr_test_nn_relu, type = "class")
test=with(bcr_test_nn_relu, table(tree.prune.pred, y))
(test[1,1]+test[2,2])/sum(test) # [1] 0.7560976


#####################
# Random forests
require(randomForest)
require(MASS)
set.seed(1)
time <- proc.time()
rf.train = randomForest(y ~ ., data = bcr_train_nn_relu, ntree = 500)
rf.train
# random forest performance (note: not fixed)
(acc_RF_relu = sum((predict(rf.train, bcr_test_nn_relu) == bcr_test_nn_relu$y))/nrow(bcr_test_nn_relu)) # [1] 0.7560976
print(proc.time()-time)

####### AUC for RandomForest
library(ROCR)
pdf("RF_breastcancer_nn_selected_relu_roc.pdf",width=10,height=6)
pred = prediction(as.numeric(predict(rf.train, bcr_test_nn_relu)), as.numeric(bcr_test_nn_relu$y))
# calculate probabilities for TPR/FPR for predictions
perf = performance(pred,"tpr","fpr")
(auc = performance(pred,"auc")@y.values[[1]]) # shows calculated AUC for model # [1] 0.7661591
plot(perf,colorize=FALSE, col="black") # plot ROC curve
lines(c(0,1),c(0,1),col = "gray", lty = 4 )
text(0.8,0.2, paste0('AUC=', round(auc,3), '\nACC=', round(acc_RF_relu,3)))
dev.off()

oob.err = double(ncol(bcr_train_nn_relu)-1)
test.err = double(ncol(bcr_train_nn_relu)-1)
for (mtry in 1:(ncol(bcr_train_nn_relu)-1)) {
  fit = randomForest(y ~ ., data = bcr_train_nn_relu, mtry = mtry, ntree = 500)
  oob.err[mtry] = fit$err.rate[500,1]
  pred = predict(fit, bcr_test_nn_relu)
  test.err[mtry] = sum(bcr_test_nn_relu$y != pred)/nrow(bcr_test_nn_relu)
  cat(mtry, " ")
}
oob.err
test.err

pdf("RF_breastcancer_nn_selected_relu.pdf",width=10,height=6)
plot(1:mtry, oob.err, pch = 19, col = c("black"), xlab = "Number of predictors chosen as split candidates",
     type = "b", ylab = "Misclassification Error", main = "Bagging and Random Forest Out-Of-Bag Error Plot (Using ReLU selected NN features)")
legend("topright", legend = c("OOB Error Rate"), pch = 19, col = "black")
abline(v = floor(sqrt(mtry)), col = "blue")
text(floor(sqrt(mtry)), 0.042, "Random 
     Forest", col = "blue")
abline(v = mtry, col = "red")
text(mtry, 0.042, "Bagging", col = "red")
1-test.err[which.min(oob.err)] # [1] 0.7621951
dev.off()



#####################################################################
#Boosting
library("rpart")
# 10 fold cv for boosting
set.seed(1)
training.boost <- bcr_train_nn_relu
training.boost$y=(as.numeric(training.boost$y)-1.5)*2
dim(training.boost)
ncv <- 10
n <- nrow(training.boost)
ll1=sample(rep(1:ncv,n/ncv)) # sample without replacement
iter <- 500
out <- matrix(NA, nrow = ncv, ncol = iter)
for (ll in 1:ncv){
  print(ll)
  sub <- ll1!=ll # subset for training set (90% of the data)
  y=training.boost[sub,]$y
  x=training.boost[sub,-dim(training.boost)[2]]
  y_test=training.boost[!sub,]$y
  x_test=training.boost[!sub,-dim(training.boost)[2]]
  f=rep(0,length(y))
  f_test=rep(0,length(y_test))
  i=1
  while(i<=iter){
    w=exp(-y*f)
    w=w/sum(w)
    fit=rpart(y~.,x,w,method="class") # fit a weak learner on training set
    g=-1+2*(predict(fit,x)[,2]>0.5)
    g_test=-1+2*(predict(fit,x_test)[,2]>0.5)
    e=sum(w*(y*g<0))
    alpha=0.5*log ((1-e)/e)
    f=f+alpha*g
    f_test=f_test+alpha*g_test
    out[ll,i]=sum(1*f_test*y_test<0)/length(y_test)
    i=i+1
  }
}
mean.cv <- apply(out, 2, mean)
which.min(mean.cv) # 100


# Use the best number of iteration
time <- proc.time()
best.iter <- which.min(mean.cv) # 100
iter <- 500
training.boost <- bcr_train_nn_relu
training.boost$y=(as.numeric(training.boost$y)-1.5)*2
dim(training.boost)
testing.boost <- bcr_test_nn_relu
testing.boost$y=(as.numeric(testing.boost$y)-1.5)*2
dim(testing.boost)

y=training.boost$y
x=training.boost[,-dim(training.boost)[2]]
y_test=testing.boost$y
x_test=testing.boost[,-dim(testing.boost)[2]]
train_error=rep(0,iter)
test_error=rep(0,iter)
train_loss=rep(0,iter)
test_loss=rep(0,iter)
f=rep(0,length(y))
f_test=rep(0,length(y_test))
i=1
while(i<=iter){
  w=exp(-y*f)
  w=w/sum(w)
  fit=rpart(y~.,x,w,method="class")
  g=-1+2*(predict(fit,x)[,2]>0.5)
  g_test=-1+2*(predict(fit,x_test)[,2]>0.5)
  e=sum(w*(y*g<0))
  alpha=0.5*log ((1-e)/e)
  f=f+alpha*g
  f_test=f_test+alpha*g_test
  if (i == best.iter){
    f_test_relu_best.iter <- f_test
  }
  train_error[i]=sum(1*f*y<0)/length(y)
  test_error[i]=sum(1*f_test*y_test<0)/length(y_test)
  i=i+1
}
print(proc.time()-time)

test_error[which.min(mean.cv)] # [1] 0.25
(acc_boost_relu <- 1-test_error[which.min(mean.cv)]) # [1] 0.75


####### AUC for AdaBoost
library(ROCR)
pdf("AdaBoost_breastcancer_nn_selected_relu_roc.pdf",width=10,height=6)
pred = prediction(f_test_relu_best.iter, as.numeric(bcr_test_nn_relu$y))
# calculate probabilities for TPR/FPR for predictions
perf = performance(pred,"tpr","fpr")
(auc = performance(pred,"auc")@y.values[[1]]) # shows calculated AUC for model [1] 0.7932527
plot(perf,colorize=FALSE, col="black") # plot ROC curve
lines(c(0,1),c(0,1),col = "gray", lty = 4 )
text(0.8,0.2, paste0('AUC=', round(auc,3), '\nACC=', round(acc_boost_relu,3)))
dev.off()
