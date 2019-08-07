info <- c()

info['lib'] <- lapply( lib <- c( #'data.table',
  'dplyr','ggplot2','GGally','plotly'  # yung'c baseline
#  ,'trees'  # for constructing classification and regression trees
#  ,'ISLR'   # Introduction to Statictical Learning with R
#  ,'HSAUR3' # A Handbook of Statistical Analyses Using R (3rd Edition)
  # https://rdrr.io/cran/HSAUR3/
#  ,'gbm'    # Generalized Boosted Regression Models
#  ,'randomForest'
#  ,'caret'#
#  ,'e1071'
  ), library, character.only=TRUE )

system.time(df <- data.table::fread(
  info['data.source'] <- './dataset/creditcard.csv' ))

str(df)
ggcorr(df[1:100,2:8], palette="RdBu", label=TRUE)
ggplotly(ggpairs(df[1:100,2:8]))

source("./helper/yc.helper.R");
missingness(df)
mydf <- splitting(df, training=0.6, testing=0.2, holding=0.2)

# Recursive Feature Elimination Useing Caret
if (!require('caret')) install.packages('caret', dependencies=c("Depends","Suggests") ); library(caret  )

set.seed(1.1)

#---------------------------
# REMOVE REDUNDANT FEATURES
#---------------------------

# ensure the results are repeatable

library(mlbench)
data(PimaIndiansDiabetes)

# calculate correlation matrix
correlationMatrix <- cor(PimaIndiansDiabetes[,1:8])
# summarize the correlation matrix
print(correlationMatrix)
# find attributes that are highly corrected (ideally >0.75)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.5)
# print indexes of highly correlated attributes
print(highlyCorrelated)

# prepare training scheme
control <- trainControl(method="repeatedcv", number=10, repeats=3)
# train the model
model <- train(diabetes~., data=PimaIndiansDiabetes, method="lvq", preProcess="scale", trControl=control)
# estimate variable importance
importance <- varImp(model, scale=FALSE)
# summarize importance
print(importance)
# plot importance
ggplot(importance)



# define the control using a random forest selection function
control <- rfeControl(functions=rfFuncs, method="cv", number=10)
# run the RFE algorithm
results <- rfe(PimaIndiansDiabetes[,1:8], PimaIndiansDiabetes[,9], sizes=c(1:8), rfeControl=control)
# summarize the results
print(results)
# list the chosen features
predictors(results)
# plot the results
plot(results, type=c("g", "o"))
