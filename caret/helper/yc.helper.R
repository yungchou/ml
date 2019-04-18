# CARET MODELS
#if (!require('caret')) install.packages('caret'); library(caret)
#names(getModelInfo())

# GET DATASET FROM URL
# e.g. http://archive.ics.uci.edu/ml/datasets/Adult
system.time(
  census1994 <- data.table::fread(
    RCurl::getURL(
     "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    ,ssl.verifypeer=FALSE
    )
  )
); head(census1994,2)

# RENAME COLUMNS
names(census1994) <- c(
   'age','workclass','fnlwgt','education','education-num'
  ,'marital-status','occupation','relationship','race','sex'
  ,'capitatal-ain','capital-loss','hours-per-week','native-country','income'
);head(census1994,2);str(census1994)

census1994$income <- ifelse(census1994$income=='<=50K',0,1)
str(census1994)

# DUMMIFICATION
dmy <- caret::dummyVars('~.',census1994) # Dummify all factor variables



missingness <- function(df, logging=FALSE){

  if (logging) cat('\nEntering the function, missingness\n')

  #-------------
  # Missingness
  #-------------
  if( any(is.na(df)) ){
    if (!require('naniar')) install.packages('naniar'); library(naniar)
    jpeg('missingness.jpg')
    gg_miss_upset(df)
    dev.off()

    #------------------------------
    # Percentage of missing values
    #------------------------------
    missing <- function(x) { round(( sum(is.na(x))/length(x) )*100, 2) }
    apply(df,2,missing)

    # Handle missing data here

  } else {
    cat('The imported dataset has no missing data.')
  }

  if (logging) cat('\nLeaving the function, missingness\n')

}


splitting <- function(df, training=0.6, testing=0.2, holding=0.2, logging=FALSE){

  if (logging) cat('\nEntering the function, splitting')

if( (training+testing+holding)!=1 ){
  cat('\nSpecified portions:','\nTraining(',training, ') + Testing(',testing, ') + Holding(', holding, ') =',
 training+testing+holding,
'\nDO NOT SUM UP TO 1!', '\nUsing training = 0.6, testgin = 0.2, and holding = 0.2, instead.\n')
  training = 0.6 ; testing = 0.2 ; holding = 0.2
}

#----------------
# Splitting data
#----------------

if (!require('dplyr')) install.packages('dplyr'); library(dplyr)

set.seed(0-0)

hold.data <- sample_frac( df, holding )

# Excluding the hold data from the imported dataset
temp <- setdiff(df, hold.data)

part <- sample(2 , nrow(temp) ,replace=TRUE ,prob=c(
  training/(training+testing),
  testing/(training+testing)
  ))

train.data <- temp[part==1,]
test.data  <- temp[part==2,]

cat('\nImported dataset =',nrow(df),'rows',
    '\nTrain portion (',training,') =',nrow(train.data),
    '\nTest  portion (',testing ,') =',nrow(test.data),
    '\nHold  portion (',holding ,') =',nrow(hold.data))

  if (logging) cat('\nLeaving the function, splitting')

return ( res <- list('train'=train.data, 'test'=test.data, 'hold'=hold.data) )

}