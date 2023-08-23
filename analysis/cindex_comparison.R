library(survival);library(survminer);library(ggplot2);library(svglite);library(tidyverse);library(dplyr);require(plyr);library(caret);library(glmnet);library(rms)
library(Hmisc)

workingDir = "."
random_seed = 42
set.seed(random_seed)
setwd(workingDir) 

# task = "dfs_align"
task = "os_align"
method_list <- list("ajcc.csv","bio.csv", "mp.csv", "rcb.csv")

load_data <- function(x){
  alldata <- read.table(x, header = T,check.names=F, sep=",") 
  label <- data.frame(alldata[, c('event', 'time')])
  data_value <- alldata[ , -which(colnames(alldata) %in% c('', 'Patient code', 'event', 'time'))]
  
  VarNames <- colnames(data_value)
  
  event = label$event
  event = gsub('True', 1, event, fixed=TRUE)
  event = gsub('False', 0, event, fixed=TRUE)
  event = as.numeric(event) 
  label[, 'event'] <- event
  alldata <- list(data_value, label)
  return (alldata)
}

for (method in method_list){
  print(method)
  # load data
  train_data <- load_data(paste(task, paste("train", method, sep="_"), sep = "\\"))
  train_value <- train_data[[1]]
  train_label <- train_data[[2]]
  val_data <- load_data(paste(task, paste("val", method, sep="_"), sep = "\\"))
  val_value <- val_data[[1]]
  val_label <- val_data[[2]]
  sx_data <- load_data(paste(task, paste("sx", method, sep="_"), sep = "\\"))
  sx_value <- sx_data[[1]]
  sx_label <- sx_data[[2]]
  lz_data <- load_data(paste(task, paste("lz", method, sep="_"), sep = "\\"))
  lz_value <- lz_data[[1]]
  lz_label <- lz_data[[2]]

  train_cdex0 = rcorr.cens(train_value, Surv(train_label$time, train_label$event))
  train_cindex = 1-train_cdex0[[1]]
  train_cindex_upper <- train_cdex0[3]/2*1.96+train_cindex
  train_cindex_lower <- train_cindex-train_cdex0[3]/2*1.96

  val_cdex0 = rcorr.cens(val_value, Surv(val_label$time, val_label$event))
  val_cindex = 1-val_cdex0[[1]]
  val_cindex_upper <- val_cdex0[3]/2*1.96+val_cindex
  val_cindex_lower <- val_cindex-val_cdex0[3]/2*1.96
  
  sx_cdex0 = rcorr.cens(sx_value, Surv(sx_label$time, sx_label$event))
  sx_cindex = 1-sx_cdex0[[1]]
  sx_cindex_upper <- sx_cdex0[3]/2*1.96+sx_cindex
  sx_cindex_lower <- sx_cindex-sx_cdex0[3]/2*1.96
  
  lz_cdex0 = rcorr.cens(lz_value, Surv(lz_label$time, lz_label$event))
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
  file_name = paste(task, paste("cindex", method, sep="_"), sep = "\\")
  write.csv(data, file = file_name, quote = F, row.names = F)
  
}