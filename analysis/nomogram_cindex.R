library(survival);library(survminer);library(ggplot2);library(svglite);library(tidyverse);library(dplyr);require(plyr);library(caret);library(glmnet);library(rms)
library(Hmisc)

workingDir = "."
random_seed = 42
set.seed(random_seed)
setwd(workingDir) 

task = "OS"
result_path_list <- list("multimodal\\result_os_experiments",
                         "table\\result",
                         "multimodal_align\\result",
                         "table_align\\result")

# task = "DFS"
# result_path_list <- list("multimodal\\result_save_dir",
#                          "multimodal_align\\result_align",
#                          "table\\result")


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
  
  feature<-completeUnivar$characteristics[completeUnivar$P.value<0.05]
  train_value_selected <- train_value[,feature]
  val_value_selected <- val_value[,feature]
  sx_value_selected = sx_value[,feature]
  lz_value_selected = lz_value[,feature]
  
  
  lasso_cox <- cv.glmnet(as.matrix(train_value_selected),y,family = "cox",type.measure = "deviance",alpha=1,nfolds = 5) #alpha=1 equals to lasso
  lasso_selected_features_index <- which(coef(lasso_cox,s="lambda.1se")!=0)
  lasso_selected_features <- colnames(train_value_selected)[lasso_selected_features_index]
  file_name = paste(task, result_path, "lasso_cox.Rdata", sep = "\\")
  save(lasso_cox, file=file_name)
  
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
  
  train_fp <- predict(f, train_value_selected[lasso_selected_features_index])
  train_cdex0 = rcorr.cens(train_fp, Surv(train_label$time, train_label$event))
  train_cindex = 1-train_cdex0[[1]]
  train_cindex_upper <- train_cdex0[3]/2*1.96+train_cindex
  train_cindex_lower <- train_cindex-train_cdex0[3]/2*1.96
    
  val_fp <- predict(f, val_value_selected[lasso_selected_features_index])
  val_cdex0 = rcorr.cens(val_fp, Surv(val_label$time, val_label$event))
  val_cindex = 1-val_cdex0[[1]]
  val_cindex_upper <- val_cdex0[3]/2*1.96+val_cindex
  val_cindex_lower <- val_cindex-val_cdex0[3]/2*1.96
  
  sx_fp <- predict(f, sx_value_selected[lasso_selected_features_index])
  sx_cdex0 = rcorr.cens(sx_fp, Surv(sx_label$time, sx_label$event))
  sx_cindex = 1-sx_cdex0[[1]]
  sx_cindex_upper <- sx_cdex0[3]/2*1.96+sx_cindex
  sx_cindex_lower <- sx_cindex-sx_cdex0[3]/2*1.96
  
  lz_fp <- predict(f, lz_value_selected[lasso_selected_features_index])
  lz_cdex0 = rcorr.cens(lz_fp, Surv(lz_label$time, lz_label$event))
  lz_cindex = 1-lz_cdex0[[1]]
  lz_cindex_upper <- lz_cdex0[3]/2*1.96+lz_cindex
  lz_cindex_lower <- lz_cindex-lz_cdex0[3]/2*1.96
  
  datanames = c("train_cindex", "train_cindex_lower", "train_cindex_upper", 
              "val_cindex", "val_cindex_lower", "val_cindex_upper",
              "sx_cindex", "sx_cindex_lower", "sx_cindex_upper",
              "lz_cindex", "lz_cindex_lower", "lz_cindex_upper")
  data = c(train_cindex, train_cindex_lower, train_cindex_upper, 
              val_cindex, val_cindex_lower, val_cindex_upper,
              sx_cindex, sx_cindex_lower, sx_cindex_upper,
              lz_cindex, lz_cindex_lower, lz_cindex_upper)
  data <- data.frame(datanames, data)
  file_name = paste(task, result_path, "Nomogram_cindex.csv", sep = "\\")
  write.csv(data, file = file_name, quote = F, row.names = F)
  
}