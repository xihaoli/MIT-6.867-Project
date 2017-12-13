####################
## Generating AUC plots
## Xihao Li
####################

library(ROCR)

lasso_boosting_prob <- read.table("~/Dropbox/6.867project/data/lasso_boosting_prob.txt")

en_boosting_prob <- read.table("~/Dropbox/6.867project/data/en_boosting_prob.txt")

pca_boosting_prob <- read.table("~/Dropbox/6.867project/data/pca_boosting_prob.txt")

svm_prob <- read.csv("~/Dropbox/6.867project/data/svm_lr_prob.csv")
svm_lasso_prob <- svm_prob$svm.prob.lasso

relu_lasso_prob <- read.table("~/Dropbox/6.867project/data/relu_lasso_prob.txt")

relu_boosting_prob <- read.table("~/Dropbox/6.867project/data/relu_boosting_prob.txt")

bcr_test_relu <- read.table("~/Dropbox/6.867project/data/nn_intermediate_relu_test.txt",header=FALSE)

label <- bcr_test_relu$V1001

pred_lasso_boosting <- prediction(lasso_boosting_prob,as.numeric(label))
perf_lasso_boosting <- performance(pred_lasso_boosting,measure = "tpr",x.measure = "fpr")
(auc_lasso_boosting = performance(pred_lasso_boosting,"auc")@y.values[[1]])

pred_en_boosting <- prediction(en_boosting_prob,as.numeric(label))
perf_en_boosting <- performance(pred_en_boosting,measure = "tpr",x.measure = "fpr")
(auc_en_boosting = performance(pred_en_boosting,"auc")@y.values[[1]])

pred_pca_boosting <- prediction(pca_boosting_prob,as.numeric(label))
perf_pca_boosting <- performance(pred_pca_boosting,measure = "tpr",x.measure = "fpr")
(auc_pca_boosting = performance(pred_pca_boosting,"auc")@y.values[[1]])

pred_svm_lasso <- prediction(svm_lasso_prob,as.numeric(label))
perf_svm_lasso <- performance(pred_svm_lasso,measure = "tpr",x.measure = "fpr")
(auc_svm_lasso = performance(pred_svm_lasso,"auc")@y.values[[1]])

pred_relu_lasso <- prediction(relu_lasso_prob,as.numeric(label))
perf_relu_lasso <- performance(pred_relu_lasso,measure = "tpr",x.measure = "fpr")
(auc_relu_lasso = performance(pred_relu_lasso,"auc")@y.values[[1]])

pred_relu_boosting <- prediction(relu_boosting_prob,as.numeric(label))
perf_relu_boosting <- performance(pred_relu_boosting,measure = "tpr",x.measure = "fpr")
(auc_relu_boosting = performance(pred_relu_boosting,"auc")@y.values[[1]])

###### Plot
pdf("AUC.pdf", height = 8, width = 8)
plot(perf_lasso_boosting, col = "purple", main = "ROC curves comparing different classifiers", lwd = 3)
plot(perf_en_boosting, col = "blue", add = TRUE, lwd = 3)
plot(perf_pca_boosting, add = TRUE, col = "forestgreen", lwd = 3)
plot(perf_svm_lasso, add = TRUE, col = "orange", lwd = 3)
plot(perf_relu_lasso, add = TRUE, col = "red", lwd = 3)
plot(perf_relu_boosting, add = TRUE, col = "magenta", lwd = 3)
abline(0,1)
legend("bottomright", c("LASSO+Boosting (AUC = 0.758)", "Elastic Net+Boosting (AUC = 0.760)", "PCA+Boosting (AUC = 0.762)", "LASSO+SVM (AUC = 0.743)", "Hidden Feature+LASSO (AUC = 0.817)", "Hidden Feature+Boosting (AUC = 0.793)"), 
       col = c("purple", "blue", "forestgreen", "orange", "red", "magenta"), lwd = c(3,3,3,3,3,3))
dev.off()

