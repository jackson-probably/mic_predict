### install necessary packges
install.packages("ggplot2")
install.packages("tidyverse")
install.packages("viridis")
install.packages("shapviz")

### load packages
library(ggplot2)
library(tidyverse)
library(viridis)
library(shapviz)

### set working directory, yours will certainly be different !!
getwd()
setwd("C:/Users/jwh8777/OneDrive - University of Illinois Chicago/Desktop/lab_materials/R")

### upload and format data, I use long data formats for basically everything
### This will be your train and test eval for each iteration in each fold

ml_data <- read_csv("data/ten_fold_df.csv")
ml_long <- ml_data %>% 
  pivot_longer(cols = 'train_rmse':'eval_rmse',
               names_to = "test",
               values_to = "rmse")

### These are labeled based on my variable names from the model.py code
shap_df <- read_csv("data/shap_values.csv")
X <- read_csv("data/X_test_for_shap.csv")
shap_matrix <- as.matrix(shap_df)

### And then this is just for the shapviz program so we can make beeswarm and feature importance figures
sv <- shapviz(shap_matrix, X=X)

### this is the test vs train plot for each fold
### of course, feel free to change colors and visual elements as you see fit, I have my own preferences

ggplot(data = ml_long, aes(x = iteration, y = rmse, 
                                  color = test,))+
  geom_point(size = 1)+
  geom_line(size = .9,)+
  scale_colour_manual(values = c("#621708",
                                 "#F6AA1C")) +
  scale_x_continuous(breaks=seq(0,300,by=50)) +
  scale_y_continuous(breaks=seq(0,3.6,by=1)) +
  ggtitle("XGBoost prediction for MIC based on 8mers") +
  ylab("Root Mean Square Error") +
  xlab(expression("Iteration")) +
  facet_wrap(~factor(fold), ncol = 5) +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5),
        aspect.ratio = 1,
        axis.title.x = element_text(size=15, face="bold", colour = "black"),    
        axis.title.y = element_text(size=15, face="bold", colour = "black"),)
        #panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        #panel.background = element_blank(), axis.line = element_line(colour = "black"))

### You can adjust these ratios as you see fit for your final figure, this is just what works for me!!!
### ggsave will not tell you if you're overwriting something so double check your file name!
ggsave("XGBoost_results.jpg",
       plot = last_plot(),   
       width = 12,           
       height = 6,           
       dpi = 300)


### and then this is the shap score visualizers
### for ggsave, it will just save your last plot
### you can also put this in a for loop, I'm just lazy and it doesn't really save that much time tbh
### that said, make sure you rename your ggsave title every time so it doesn't overwrite your figures!

sv_importance(sv)
sv_beeswarm(sv)
sv_importance(sv, kind = "beeswarm")

ggsave("mean_shap_plot.jpg",
       plot = last_plot(),   
       width = 12,           
       height = 8,           
       dpi = 1000)

