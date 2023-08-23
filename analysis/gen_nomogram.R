library(survival);library(survminer);library(ggplot2);library(svglite);library(tidyverse);library(dplyr);require(plyr);library(caret);library(glmnet);library(rms)


workingDir = "."
random_seed = 42
set.seed(random_seed)
setwd(workingDir) 

task = "OS"
result_path_list <- list("multimodal\\result_os",
                         "table\\result_os",
                         "multimodal_align\\result_os")

# task = "DFS"
# result_path_list <- list("multimodal\\result",
                         # "multimodal_align\\result_align",
                         # "table\\result")


load_data <- function(x){
  alldata <- read.table(x, header = T,check.names=F, sep=",") 
  label <- data.frame(alldata[, c('event', 'time', 'prediction')])
  data_value <- alldata[ , -which(colnames(alldata) %in% c('', 'Patient code', 'event', 'time', 'prediction'))]
  
  VarNames <- colnames(data_value)
  
  for (varName in VarNames){
    varName_new <- gsub(" ","",varName)
    varName_new <- gsub("-","_",varName_new)
    varName_new <- gsub(":","_",varName_new)
    names(data_value)[names(data_value) == varName] <- varName_new
    
    if (grepl("_0", varName_new) | grepl("pCR_1", varName_new)){
      data_value <- data_value[ , -which(colnames(data_value) %in% c(varName_new))]
    }
  }
  
  event = label$event
  event = gsub('True', 1, event, fixed=TRUE)
  event = gsub('False', 0, event, fixed=TRUE)
  event = as.numeric(event) 
  label[, 'event'] <- event
  alldata <- list(data_value, label)
  return (alldata)
}

for (result_path in result_path_list){
  print(result_path)
  # load data
  train_data <- load_data(paste(task, result_path, "train.csv", sep = "\\"))
  train_value <- train_data[[1]]
  train_label <- train_data[[2]]
  val_data <- load_data(paste(task, result_path, "val.csv", sep = "\\"))
  val_value <- val_data[[1]]
  val_label <- val_data[[2]]
  sx_data <- load_data(paste(task, result_path, "sx.csv", sep = "\\"))
  sx_value <- sx_data[[1]]
  sx_label <- sx_data[[2]]
  lz_data <- load_data(paste(task, result_path, "lz.csv", sep = "\\"))
  lz_value <- lz_data[[1]]
  lz_label <- lz_data[[2]]
  
  y <- Surv(train_label$time, train_label$event==1)
  
  Unicox.result<-function(x){
    FML<-as.formula(paste0("y~",x))
    unicox<-coxph(FML,ties=c("breslow"),data=train_value)
    unicox.sum<-summary(unicox)
    CI<-paste0(round(unicox.sum$conf.int[,3:4],2),collapse = "-")
    HR<-round(unicox.sum$coefficients[,2],2)
    Pvalue<-round(unicox.sum$coefficients[,5],4)
    unicox.result<-data.frame("characteristics"=x,
                              "Hazard Ratio"=HR,
                              "CI95"=CI,
                              "P value"=Pvalue)
    return(unicox.result)
  }
  
  VarNames <- colnames(train_value)
  Univar<-lapply(VarNames,Unicox.result)
  
  Univar<-ldply(Univar,data.frame)
  completeUnivar <- Univar[complete.cases(Univar),]
  save_path = paste(task, result_path, "univarcox.csv", sep = "\\")
  write.table(completeUnivar, save_path, sep = ",")
  
  ###Clinical+dl Signature Nomogram Building Signature nomogram
  #1 select pvalue<0.05 univariate
  feature<-completeUnivar$characteristics[completeUnivar$P.value<0.05]
  train_value_selected <- train_value[,feature]
  val_value_selected <- val_value[,feature]
  
  #2 detect the high correlation features
  highcorr <- findCorrelation(cor(train_value_selected),cutoff = 0.6,names = T)
  print('High correlated features')
  print(highcorr) #manually decide whether to remove
  
  #3 select features using lasso
  lasso_cox <- cv.glmnet(as.matrix(train_value_selected),y,family = "cox",type.measure = "deviance",alpha=1,nfolds = 5) #alpha=1 equals to lasso
  lasso_selected_features_index <- which(coef(lasso_cox,s="lambda.1se")!=0)
  lasso_selected_features <- colnames(train_value_selected)[lasso_selected_features_index]
  file_name = paste(task, result_path, "lasso_cox.Rdata", sep = "\\")
  save(lasso_cox, file=file_name)
  
  ### plot nomogram
  tr_nomogram <- train_value_selected[lasso_selected_features_index]
  dd=datadist(tr_nomogram)
  options(datadist="dd")
  formula <-as.formula(paste0("y~",paste0(lasso_selected_features,collapse = "+")))
  print('Final used formulation')
  print(formula)
  
  f <- cph(formula, ties=c('breslow'), data = tr_nomogram, surv = TRUE) 
  survival <- Survival(f)
  survival1 <-  function(x)survival(12,x) # 1 year not enough data
  survival3 <-  function(x)survival(36,x) # 3 years
  survival5 <-  function(x)survival(60,x) # 5 years
  
  nom2 <- nomogram(f, fun=list(survival3,survival5), 
                   fun.at = c(0.01,seq(.1,.9, by= .1),.95,.99),
                   funlabel=c('3 years Survival Probability','5 years Survival Probability'))
  
  file_name = paste(task, result_path, "Nomogram.pdf", sep = "\\")
  pdf(file_name,width=13, height=8)
  plot(nom2)
  dev.off()
#}