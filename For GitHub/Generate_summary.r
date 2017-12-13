#### MIT 6.867 Breast Cancer Subtyping
### Generate Final summary plot
# Helian Feng

library(readr)
library(tidyr)
library(dplyr)
library(stats)
require(ggplot2)
require(ggrepel)
require()
comparison_table <- read_csv("~/Dropbox/6.867project/comparison_table.csv")
comparison_table$Space_requirement=factor(comparison_table$Space_requirement,levels=c("Small",  "Medium" ,"Large" ),ordered = TRUE)
ggplot(comparison_table) +
  geom_point(aes(Time_scale, AUC, shape = Interpretability,size=Space_requirement))+
  scale_x_log10(limits=c(0.01, 10000)) +
  geom_label_repel(
    aes(Time_scale, AUC, fill = Dimensional_reduction_approach, label = comparison_table$Method),label.size = 0.05,
    fontface = 'bold', color = 'white',size = 4.5,
    #box.padding = 0.35, point.padding = 0.5,
    segment.color = 'grey50'
  ) + 
  #scale_fill_manual(values=c("red", "blue", "green"))+ylim(0.4,1)+
  theme_classic(base_size = 8)
