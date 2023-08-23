library(survival);library(survminer);library(ggplot2);library(svglite);library(tidyverse)

workingDir = "."
random_seed = 42
set.seed(random_seed)
setwd(workingDir) 

task = "OS"
result_path_list <- list("multimodal\\result_os",
                        "table\\result",
                        "multimodal_align\\result_align_os",
                        "table_align\\result_align_os",
                        "path")

# task = "DFS"
# result_path_list <- list("multimodal\\result_dfs",
#                          "multimodal_align\\result_dfs",
#                          "path",
#                          "table\\result_dfs",
#                          "table_align\\result_align_dfs",
#                          )
data_list <- list("train.csv", "val.csv", "sx.csv", "lz.csv")

for (result_path in result_path_list){
  for (data_split in data_list){
   data <-read.table(paste(task, result_path, data_split, sep = "\\"),header = T,check.names=F, sep=",") 
   data_clean <- data.frame(data[, c('event', 'time', 'prediction')])
   event = data_clean$event
   event = gsub('True', 1, event, fixed=TRUE)
   event = gsub('False', 0, event, fixed=TRUE)
   event = as.numeric(event)     
   plot_km <- data.frame(risk=data_clean$prediction, time=data_clean$time, event=event)
   cutpoint <- surv_cutpoint(
     plot_km,
     time = "time",
     event = "event",
     variables = "risk"
   )
   print(summary(cutpoint))
   write.table(summary(cutpoint), file = paste(task, result_path, paste("cutoff", data_split, sep="_"), sep = "\\") ,sep = ",", row.names = FALSE)

   plot_km <- surv_categorize(cutpoint)
   names(plot_km)[3] <-'risk'
   kmmodel<-survfit(Surv(time, event)~risk,data=plot_km)
   
   ggsurv <- ggsurvplot(
     kmmodel, # survfit object with calculated statistics.
     pval = TRUE,             # show p-value of log-rank test.
     conf.int = FALSE,         # show confidence intervals for point estimaes of survival curves.
     xlim = c(0,140),        # present narrower X axis, but not affect
     # survival estimates.
     break.time.by = 10,    # break X axis in time intervals by 500.
   )
   save_path = str_replace(paste(task, result_path, paste("surv", data_split, sep="_"), sep = "\\"), "csv", "pdf")
   pdf(save_path, onefile=FALSE)
   print(ggsurv)
   dev.off()
   }
}
