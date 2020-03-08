library(MASS)
library(ggplot2)
library(boot)

#Load the Boston sample dataset into R using a dataframe
data(Boston)
#Use lm to fit a regression between medv and lstat
model<-lm(medv~lstat, data=Boston)
summary(model)
# plot the resulting fit 
ggplot(Boston,aes(x=medv,y=lstat)) +
  geom_point() + geom_smooth(method=lm)
#and show a plot of fitted values vs. residuals. 
metadata<-data.frame("fit"=fitted(model),
                     "r"=resid(model))
ggplot(data = metadata)+
  geom_point(mapping = aes(x = fit,y=r))+
  geom_smooth(mapping = aes(x = fit,y=r))
#Is there a possible non-linear relationship between 
#the predictor and response?
#Yeah. That looks very non-linear.


#Use the predict function to calculate values response 
#values for lstat of 5, 10, and 15
#obtain confidence intervals as well as prediction intervals 
#for the results
predict(model,data.frame(lstat=c(5,10,15)), interval = 'confidence',level=.95)
predict(model,data.frame(lstat=c(5,10,15)), interval = 'prediction',level=.95)

#are they the same? Why or why not?
#No. The prediction interval is actually much wider. This is because it is
#taking into account the variance of the error term for new response values.


#Modify the regression to include lstat2
mod2<-lm(medv~lstat+I(lstat^2), data=Boston)
#compare the R2 between the linear and non-linear fit
summary(mod2)$r.squared
summary(model)$r.squared

#use ggplot2 and stat smooth to plot the relationship.
ggplot(Boston, aes(x = lstat, y = medv)) + 
  geom_point() + stat_smooth(formula = y~x+I(x^2), 
                             method = "lm", se= FALSE, color = "red")
#PRACTICUM PROBLEM #2
#Load the abalone sample dataset from the UCI Machine Learning 
#Repository (abalone.data) into R using a dataframe. 
library(caret)
col_names=c("Sex", "Length", "Diameter", "Height", "Whole_Weight","Shucked_Weight", "Viscera_Weight", "Shell_Weight", "Rings")
df = read.csv('https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data', header=FALSE, sep=',', col.names = col_names)
head(df)
#Remove all observations in the Infant category, keeping the Male/Female classes.
d= df[df$Sex!='I',]
d$Sex <- factor(d$Sex, labels = c("Male","Female"))

#Using the caret package, use createDataPartition to perform an 80/20 test-train split
splitdata <- createDataPartition(d$Sex, p=0.8, list=FALSE, times=1)
train <- d[ splitdata,]
test  <- d[-splitdata,]

#Fit a logistic regression using all feature variables
logfit<-glm(Sex~.,data=train,family=binomial)
summary(logfit)
#Length, Diameter, and Height are most significant.
exp(coef(logfit))
#Do the confidence intervals for the predictors 
#contain 0 within the range?
#Diameter, Height, Shucked_Weight do not contain 0.
confint(logfit)
#How does this relate to the null hypothesis?
#Since zero is in the interval, the null CANNOT be rejected for this confidence level!
#The ones without zero in the interval are significant for the regression,
# and thus are good to reject null, though.

#Use the confusionMatrix function in caret to observe testing results
#tofix
pred<-predict(logfit,newdata = test)
pred.dt<-ifelse(pred>0.50, "M","F")
Pred <- as.factor(pred.dt)
Predicted <- ordered(Pred, levels = c("M", "F"))
Actual <- ordered(test$Sex,levels = c("M", "F"))
install.packages('e1071', dependencies=TRUE, repos = "http://cran.us.r-project.org")
confusionMatrix(table(Predicted,Actual))
#how does the accuracy compare to a random classifier ROC curve?
resp <- predict(logfit, test, type = "response")
test$resp=resp
hist(resp)
head(test)
pred <- ifelse(resp > 0.5, "Male", "Female") 

#ROC:
test$Sex
test$resp
library(pROC)
ROC1 <- roc(as.numeric(test$Sex), test$resp)
plot(ROC1, col = "blue")
#We calculate the corr matrix
predictors<-d[c(-1)]
corMatrix <- cor(predictors)
library(corrplot)
corrplot(corMatrix,type = "upper", method = "circle",diag = TRUE, tl.col = "blue",tl.srt = 45,)

#Load the mushroom sample dataset from the UCI Machine Learning Repository
names <- c("edibility","cap-shape","cap-surface","cap-color","bruises","odor","gill-attachment","gill-spacing","gill-size",
           "gill- color","stalk-shape","stalk-root","stalk-surface-above-ring","stalk-surface-below-ring","stalk-color-above-ring",
           "stalk-color- below-ring", "veil-type","veil-color","ring-number","ring-type","spore-print-color","population","habitat")
mush<-read.csv("/Users/renahaswah/Desktop/Data Prep/HW/HW2/agaricus-lepiota.data",header=FALSE,col.names = names)
#Some values in stalk.root have "?" values.
mush[mush == '?'] <- NA 
num<-is.na(mush$stalk.root)
sum(num)
# Mean, Median, Mode Imputation is acceptable, but  KNN is 
#recently taught in this course, so let's do that.
library(VIM)
y<-kNN(mush,variable = colnames(mush[12]), k=5) 
y[y=='?']<-NA
anyNA(y)
summary(mush)

library(caret)
library(e1071)
library(caTools)
#Create a Naive Bayes classifier using the e1071 package, using 
#the sample func- tion to split the data between 80% for training and 20% for testing.
train_sample <- sample(8124, floor(.8*8124))
train <- mush[train_sample, ]
test  <- mush[-train_sample, ]

library(naivebayes)
nbmodl<-naiveBayes(train$edibility~., train)

#With the target class of interest being edible mushrooms, calculate the accuracy of the 
#classifier both in-training and in-test.
#Accuracy is the percentage of values the model predicted correclty.
#In training
pt<-predict(nbmodl,train,type="class")
cmmush<-table(pt, train$edibility,dnn=c("Prediction","Actual"))
n<-sum(cmmush)
dig<-diag(cmmush)
acc<-sum(dig)/n
acc
#In test
p<-predict(nbmodl, test, type = "class")
cmmush<-table(p, test$edibility,dnn=c("Prediction","Actual"))
n<-sum(cmmush)
dig<-diag(cmmush)
acc<-sum(dig)/n
acc
#Use the table function to create a con- fusion matrix of predicted vs. actual classes - 
table(p, test$edibility,dnn=c("Prediction","Actual"))
#how many false positives did the model produce?
#Let's say edible is true.
#There are 89 values that were falsely identified as true edible.









