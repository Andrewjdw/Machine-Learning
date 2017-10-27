dir <- 'http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'
wdbc.data <- read.csv(dir,header = F)
names(wdbc.data) <- c('ID','Diagnosis','radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','concave points_mean',
                      'symmetry_mean','fractal dimension_mean','radius_sd','texture_sd','perimeter_sd','area_sd','smoothness_sd','compactness_sd','concavity_sd','concave points_sd',
                      'symmetry_sd','fractal dimension_sd','radius_max_mean','texture_max_mean','perimeter_max_mean','area_max_mean','smoothness_max_mean',
                      'compactness_max_mean','concavity_max_mean','concave points_max_mean','symmetry_max_mean','fractal dimension_max_mean')
table(wdbc.data$Diagnosis) ## M = malignant, B = benign
# 将目标属性编码因子类型
wdbc.data$Diagnosis <- factor(wdbc.data$Diagnosis,levels =c('B','M'),labels = c(B = 'benign',M = 'malignant'))
wdbc.data$Diagnosis
table(wdbc.data$Diagnosis)
prop.table(table(wdbc.data$Diagnosis))*100 ## prop.table():计算table各列的占比
round(prop.table(table(wdbc.data$Diagnosis))*100,digit =2) ## 保留小数点后两位，round()：digit =2
str(wdbc.data)

#数值型数据标准化
# min-max标准化:(x-min)/(max-min)
normalize <- function(x) { return ((x-min(x))/(max(x)-min(x))) }
normalize(c(1, 3, 5)) ## 测试函数有效性
wdbc.data.min_max <- as.data.frame(lapply(wdbc.data[3:length(wdbc.data)],normalize))
wdbc.data.min_max$Diagnosis <- wdbc.data$Diagnosis
str(wdbc.data.min_max)

#划分train&test
# train
set.seed(3) ## 设立随机种子
train_id <- sample(1:length(wdbc.data.min_max$area_max_mean), length(wdbc.data.min_max$area_max_mean)*0.7)
train <- wdbc.data.min_max[train_id,] # 70%训练集
summary(train)
train_labels <- train$Diagnosis
train <- wdbc.data.min_max[train_id, - length(wdbc.data.min_max)]
summary(train)

# test
test <- wdbc.data.min_max[-train_id,]
test_labels <- test$Diagnosis
test <- wdbc.data.min_max[-train_id,-length(wdbc.data.min_max)]
summary(test)

#knn分类（欧氏距离）
library(class)
test_pre_labels <- knn(train,test,train_labels,k=7) ## 数据框，K个近邻投票,欧氏距离

#性能评估
library(gmodels)
CrossTable(x = test_labels, y = test_pre_labels, prop.chisq = F)
