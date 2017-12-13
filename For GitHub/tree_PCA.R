#### MIT 6.867 Tree-based methods with PCA
# Xihao Li

setwd("~/Dropbox/6.867project/Milestone_5/figures/")
###### Read the train set with PCA
bcr_train_pca <- read.csv("~/Dropbox/6.867project/data/pca_train.csv",header=TRUE)
dim(bcr_train_pca) # [1] 329 329
bcr_train_pca[1:5,1:10]
bcr_train_pca$y <- as.factor(bcr_train$y)
dim(bcr_train_pca) # [1] 329 330
sum(bcr_train_pca$y == 1) # 169
sum(bcr_train_pca$y == 0) # 160
dim(bcr_train_pca)

pca_select <- read.csv("~/Dropbox/6.867project/data/pca_select.csv",header=TRUE,stringsAsFactors = FALSE)
dim(pca_select)
pca.list <- colnames(pca_select)

bcr_train_pca <- bcr_train_pca[,colnames(bcr_train_pca) %in% pca.list]
bcr_train_pca$y <- bcr_train$y
dim(bcr_train_pca)

###### Read the test set
bcr_test_pca <- read.csv("~/Dropbox/6.867project/data/pca_test.csv",header=TRUE)
dim(bcr_test_pca) # [1] 164 329
bcr_test_pca[1:5,1:10]
bcr_test_pca$y <- as.factor(bcr_test$y)
dim(bcr_test_pca) # [1] 164 330
sum(bcr_test_pca$y == 1) # 77
sum(bcr_test_pca$y == 0) # 87
dim(bcr_test_pca)

bcr_test_pca <- bcr_test_pca[,colnames(bcr_test_pca) %in% pca.list]
bcr_test_pca$y <- bcr_test$y
dim(bcr_test_pca)

###############
# Tree based methods
###############
library(tree)

tree.single <- tree(y ~ ., data = bcr_train_pca)
summary(tree.single)

pdf("Tree_single_pca.pdf",width = 10, height = 6)
plot(tree.single, main = "Single Decision Tree (Using selected PCs)")
text(tree.single, pretty = 0)
dev.off()

# Print a detailed summary of the tree
tree.single

tree.pred = predict(tree.single, newdata = bcr_test_pca, type = "class") # Note: there is randomness here
(test=with(bcr_test_pca, table(tree.pred, y)))

(test[2,2] + test[1,1])/sum(test) # [1] 0.6463415

# Pruning
set.seed(3)
bcr.tree.cv = cv.tree(tree.single, FUN = prune.misclass) # use number of misclassifications as the objective function
bcr.tree.cv # size is the total number of leaf nodes, k is the cost-complexity parameter


min_idx = which.min(bcr.tree.cv$dev) # The first index with smallest dev
bcr.tree.cv$size[min_idx]
prune.train = prune.misclass(tree.single, best = bcr.tree.cv$size[min_idx])

pdf("Tree_prune_pca.pdf",width=10,height=6)
plot(prune.train, main = "Pruned Decision Tree (Using selected PC)")
text(prune.train, pretty = 0)
dev.off()

tree.prune.pred = predict(prune.train, newdata = bcr_test_pca, type = "class")
test=with(bcr_test_pca, table(tree.prune.pred, y))
(test[1,1]+test[2,2])/sum(test) # [1] 0.6280488


#####################
# Random forests
require(randomForest)
require(MASS)
set.seed(0)
time <- proc.time()
rf.train = randomForest(y ~ ., data = bcr_train_pca, ntree = 500)
rf.train
# random forest performance
(acc_RF_pca = sum((predict(rf.train, bcr_test_pca) == bcr_test_pca$y))/nrow(bcr_test_pca)) # [1] 0.6707317
print(proc.time()-time)

####### AUC for RandomForest
library(ROCR)
pdf("RF_breastcancer_pca_roc.pdf",width=10,height=6)
pred = prediction(as.numeric(predict(rf.train, bcr_test_pca)), as.numeric(bcr_test_pca$y))
# calculate probabilities for TPR/FPR for predictions
perf = performance(pred,"tpr","fpr")
(auc = performance(pred,"auc")@y.values[[1]]) # shows calculated AUC for model # [1] 0.6834602
plot(perf,colorize=FALSE, col="black") # plot ROC curve
lines(c(0,1),c(0,1),col = "gray", lty = 4 )
text(0.8,0.2, paste0('AUC=', round(auc,3), '\nACC=', round(acc_RF_pca,3)))
dev.off()

oob.err = double(ncol(bcr_train_pca)-1)
test.err = double(ncol(bcr_train_pca)-1)
for (mtry in 1:(ncol(bcr_train_pca)-1)) {
  fit = randomForest(y ~ ., data = bcr_train_pca, mtry = mtry, ntree = 500)
  oob.err[mtry] = fit$err.rate[500,1]
  pred = predict(fit, bcr_test_pca)
  test.err[mtry] = sum(bcr_test_pca$y != pred)/nrow(bcr_test_pca)
  cat(mtry, " ")
}
oob.err
test.err

pdf("RF_breastcancer_pca.pdf",width=10,height=6)
plot(1:mtry, oob.err, pch = 19, col = c("black"), xlab = "Number of predictors chosen as split candidates",
     type = "b", ylab = "Misclassification Error", main = "Bagging and Random Forest Out-Of-Bag Error Plot (Using selected PCs)")
legend("topright", legend = c("OOB Error Rate"), pch = 19, col = "black")
abline(v = floor(sqrt(mtry)), col = "blue")
text(floor(sqrt(mtry)), 0.24, "Random 
     Forest", col = "blue")
abline(v = mtry, col = "red")
text(mtry, 0.242, "Bagging", col = "red")
1-test.err[which.min(oob.err)] # [1] 0.6463415
dev.off()



#####################################################################
#Boosting
library("rpart")
# 10 fold cv for boosting
set.seed(1)
training.boost <- bcr_train_pca
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
which.min(mean.cv) # 407


# Use the best number of iteration
time <- proc.time()
best.iter <- which.min(mean.cv) # 407
iter <- 500
training.boost <- bcr_train_pca
training.boost$y=(as.numeric(training.boost$y)-1.5)*2
dim(training.boost)
testing.boost <- bcr_test_pca
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
    f_test_pca_best.iter <- f_test
  }
  train_error[i]=sum(1*f*y<0)/length(y)
  test_error[i]=sum(1*f_test*y_test<0)/length(y_test)
  i=i+1
}
print(proc.time()-time)

test_error[which.min(mean.cv)] # [1] 0.3231707
(acc_boost_pca <- 1-test_error[which.min(mean.cv)]) # [1] 0.6768293


####### AUC for AdaBoost
library(ROCR)
pdf("AdaBoost_breastcancer_pca_roc.pdf",width=10,height=6)
pred = prediction(f_test_pca_best.iter, as.numeric(bcr_test_pca$y))
# calculate probabilities for TPR/FPR for predictions
perf = performance(pred,"tpr","fpr")
(auc = performance(pred,"auc")@y.values[[1]]) # shows calculated AUC for model # [1] 0.7617555
plot(perf,colorize=FALSE, col="black") # plot ROC curve
lines(c(0,1),c(0,1),col = "gray", lty = 4 )
text(0.8,0.2, paste0('AUC=', round(auc,3), '\nACC=', round(acc_boost_pca,3)))
dev.off()
