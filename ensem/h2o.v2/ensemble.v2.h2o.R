{ # Custom Settings
  i2 <- c() # information of h2o

  percentage <- i2['percentage'] <- 0.1  # Percentage of imported data used
  use.smote  <- i2['use.smote']  <- TRUE # For class imbalance
  train.part <- i2['train.part'] <- 0.6  # train, valid=test=(1-train)/2
  nfolds     <- i2['nfolds']     <- 10    #  nfolds cross-vlidation

  prefix        <- i2['prefix']         <- 'SL' # The heading of all output
  output.folder <- i2['output.folder' ] <- 'result/'

  saveRDS(i2,'i2.rds')

  set.seed(seed <- 55) # Initializing a default seed value
}

{ # LIBRARIES
  if (!require('filesstrings')) install.packages('filesstrings'); library(filesstrings)
  if (!require('dplyr')) install.packages('dplyr'); library(dplyr)
  if (!require('plotly')) install.packages('plotly'); library(plotly)

  if (!require('munsell')) install.packages("munsell"); library(munsell)
  if (!require('ggplot2')) install.packages('ggplot2'); library(ggplot2)

  if (!require('SuperLearner')) install.packages('SuperLearner'); library(SuperLearner)
  if (!require('ranger' )) install.packages('ranger' ); library(ranger )
  if (!require('xgboost')) install.packages('xgboost'); library(xgboost)
  if (!require('caret'  )) install.packages('caret'  ); library(caret  )

  if(FALSE){  # OPTIONAL

    if (!require('nnet')) install.packages('nnet'); library(nent)
    if (!require('NeuralNetTools')) install.packages('NeuralNetTools'); library(NeuralNetTools)

    if (!require('e1071')) install.packages('e1071'); library(e1071)
    if (!require('party')) install.packages('party'); library(party)
    if (!require('gam')) install.packages('gam'); library(gam)
    if (!require('LogicReg')) install.packages('LogicReg'); library(LogicReg)
    if (!require('polspline')) install.packages('polspline'); library(polspline)
    if (!require('extraTrees')) install.packages('extraTrees'); library(extraTrees)
    if (!require('biglasso')) install.packages('biglasso'); library(biglasso)
    if (!require('dbarts')) install.packages('dbarts'); library(dbarts)
    if (!require('speedglm')) install.packages('speedglm'); library(speedglm)
    if (!require('mlbench')) install.packages('mlbench'); library(mlbench)
    if (!require('rpart')) install.packages('rpart'); library(rpart)

    #install.packages("devtools");library(devtools
    #);install_github("AppliedDataSciencePartners/xgboostExplainer",force = TRUE)

    if (!require('MASS')) install.packages('MASS'); library(MASS)
    if (!require('cvAUC')) install.packages('cvAUC'); library(cvAUC)
    if (!require('kernlab')) install.packages('kernlab'); library(kernlab)
    if (!require('arm')) install.packages('arm'); library(arm)
    if (!require('ipred')) install.packages('ipred'); library(ipred)
    if (!require('KernelKnn')) install.packages('KernelKnn'); library(KernelKnn)
    if (!require('RcppArmadillo')) install.packages('RcppArmadillo'); library(RcppArmadillo)

  }

}

{ # DATA SET

  df.imported <- read.csv( imported.file <- 'data/capstone.dataimp.csv' )[-1]
  df.imported.obs <- nrow(df.imported)
  df.imported <- sample_frac(df.imported, percentage, replace=FALSE)

  # Set label as factor
  df.imported$readmitted <- as.factor(df.imported$readmitted)

  # PARTITIONING DATA

  # Instead of just sampling directly from df.import, which
  # may potentially include very few under-sampling class, here 1's,
  # the following ensures there are proportional o's and 1's
  # included in training, testing and cross-validation data.

  valid.part <- test.part <- (1-train.part)/2

  zeropart <- sample(3, nrow( df.imported[df.imported$readmitted==0,]),
                  replace=TRUE, prob=c(train.part, valid.part, test.part))
  onepart  <- sample(3, nrow( df.imported[df.imported$readmitted==1,]),
                  replace=TRUE, prob=c(train.part, valid.part, test.part))

  # Ensure both zeros and ones are sampled for each partition
  train.org <- rbind( df.imported[zeropart==1,] , df.imported[onepart==1,] )
  valid     <- rbind( df.imported[zeropart==2,] , df.imported[onepart==2,] )
  test      <- rbind( df.imported[zeropart==3,] , df.imported[onepart==3,] )
}

{# CLASS IMBALANNCE
  if(use.smote){ # METHOD 1: USING SMOTE TO REBALANCE TEE LABEL
    if (!require('DMwR')) install.packages('DMwR'); library(DMwR)

    train <- SMOTE(readmitted~., train.org,
                   perc.over=400, perc.under=200)

  } else { # METOD 2: SAMPLING EUQAL NUMBER OF 0's and 1's FROM THE LABEL
    n <- min(train.org[train.org$readmitted==1,],train.org[train.org$readmitted==0,])
    train <- train.balanced <- rbind(
      sample(n,train.org[train.org$readmitted==0,], replace=FALSE)
      ,sample(n,train.org[train.org$readmitted==1,], replace=FALSE)
    )
  }

  table(train.org$readmitted)
  table(train$readmitted)

  par(mfrow=c(1,2)
  );plot(train.org['readmitted'] ,las=1 ,col='lightblue' ,xlab='label(readmitted)'
         ,main=sprintf('Class Imbalance of Imported \nTraining Label\n(%i vs. %i)'
                       ,table(train.org$readmitted)[1],table(train.org$readmitted)[2])
  );plot(train['readmitted'] ,las=1 ,col='lightgreen' ,xlab='label(readmitted)'
         ,main=sprintf("SMOTE'd/Employed\nTraining Label\n(%i vs. %i)"
                       ,table(train$readmitted)[1],table(train$readmitted)[2])
  );par(mfrow=c(1,1))
}

{ # THE DISTRIBUTION OF LABEL VALUES IN EACH PARTITION OF THE EMPLOYED DATA
  label.dist <- rbind(summary(train$readmitted), summary(test$readmitted))
  label.dist <- cbind(label.dist,c(nrow(train), nrow(test)))
  rownames(label.dist) <- c('train','test')
  colnames(label.dist) <- c('0','1','Total.obs')
  label.dist
}

{ # H2O INITIATION
  if (!require('h2o')) install.packages('h2o'); library(h2o)
  if (!require('cvAUC')) install.packages('cvAUC'); library(cvAUC)
  #if (!require('h2oEnsemble')) install.packages(
   # "https://h2o-release.s3.amazonaws.com/h2o-ensemble/R/h2oEnsemble_0.1.8.tar.gz"
    #, repos = NULL); library(h2oEnsemble)

  if (!require('h2oEnsemble')){
    library(devtools)
    install_github("h2oai/h2o-3/h2o-r/ensemble/h2oEnsemble-package")
  };library(h2oEnsemble)

  # h2o initialization
  my.local.h2o <- h2o.init(ip='localhost',port=13579, nthreads=-1)
  h2o.removeAll()

  #h2o.clusterInfo()
  #demo(h2o.gbm)
  #h2o.shutdown()
}

{ # CONVERTING DATA PARTITIONS TO H2O OBJECTS
  training_frame   <- as.h2o(train)
  validation_frame <- as.h2o(valid)
  testing_frame    <- as.h2o(test)

  # SETTING UP THE LABEL
  y <- 'readmitted'
  x <- setdiff(names(training_frame), y)
}

{ # SETTING UP OUTPUT FILE PATH
  prefix   <- i2['prefix']   <- paste0('SL', nrow(train.org))
  save.dir <- i2['save.dir'] <- paste0(output.folder,prefix,'.')

  saveRDS(training_frame  ,paste0(save.dir,'training_frame.rds'))
  saveRDS(validation_frame,paste0(save.dir,'validation_frame.rds'))
  saveRDS(testing_frame   ,paste0(save.dir,'testing_frame.rds'))
}

{ # ENSEMBLE SETTINGS
  family <- c(AUTO='AUTO',binomial='binomial',guasian='guassian')
#  learner <- c(
#     gbm='h2o.gbm.wrapper'
#    ,glm='h2o.glm.wrapper'
#    ,rf ='h2o.randomForest.wrapper'
#    ,dl='h2o.deeplearning.wrapper'
#  )
}

if (skip <- TRUE) {

  #h2o.shutdown()
  { # MODEL FITTING
    fit <- h2o.ensemble( x=x, y=y, seed=seed, learner=learner
     ,training_frame = training_frame, validation_frame = validation_frame
     ,cvControl = list(V=nfolds, shuffle=TRUE)
     ,family= family['binomial'] ,model_id='ensemble' ,metalearner=learner['rf']
    );fit
  }

    perf <- h2o.ensemble_performance(fit, newdata=testing_frame);perf
    #plot(perf)

    #perf$fit@metrics$AUC
    #print(perf, metric = "MSE")
    pred <- predict.h2o.ensemble(fit, newdata=testing_frame)

    # Confusion Matrix
    perf$ensemble@metrics$cm

    # F1 score

    #AUC
    labels <- as.data.frame(validation_frame[,c(y)])[,1]
    # Ensemble test AUC
    AUC(predictions=as.data.frame(pred$pred)[,1], labels=labels)
    L <- length(learner)
    sapply(seq(L), function(l) AUC(predictions = as.data.frame(pred$basepred)[,l], labels = labels))

    model <- lapply(fit@model_ids,h2o.getModel)

}

{ #-----------------------
  # LEARNERS WITH DEFAULT
  #-----------------------
  rf <- h2o.randomForest( x, y, model_id='rf' ,nfolds=nfolds ,seed=seed
    ,training_frame = training_frame, validation_frame = validation_frame
    ,fold_assignment='Modulo', keep_cross_validation_predictions=TRUE)
  gbm <- h2o.gbm(x, y,model_id='gbm',nfolds=nfolds ,seed=seed
    ,training_frame = training_frame, validation_frame = validation_frame
    ,fold_assignment='Modulo', keep_cross_validation_predictions=TRUE)
  glm <- h2o.glm(x, y, ,model_id='glm',nfolds=nfolds ,seed=seed ,family= family['binomial']
    ,training_frame = training_frame, validation_frame = validation_frame
    ,fold_assignment='Modulo', keep_cross_validation_predictions=TRUE)
  dl <- h2o.deeplearning(x, y, model_id='dl',nfolds=nfolds ,seed=seed
    ,training_frame = training_frame, validation_frame = validation_frame
    ,fold_assignment='Modulo', keep_cross_validation_predictions=TRUE)

  models <- list(rf@model_id, gbm@model_id, glm@model_id, dl@model_id)
  learners <- c(rf, gbm, glm, dl);saveRDS(learners,paste0(save.dir,'learners.rds'))

}

if(skip <- TRUE){

  h2o.confusionMatrix(rf,  valid=FALSE)
  h2o.confusionMatrix(gbm, valid=FALSE)
  h2o.confusionMatrix(glm, valid=FALSE)
  h2o.confusionMatrix(dl,  valid=FALSE)

  summary(rf)
  rf@model$model_summary
  rf@model$scoring_history
  rf@model$training_metrics
  rf@model$training_metrics@metrics$cm$table
  rf@model$variable_importances
}

{ #-----------------
  # STACKED ENSEMBLE
  #-----------------
  stacked <- h2o.stackedEnsemble(x, y, seed=seed
      ,model_id='stacked',base_models=models
      ,training_frame = training_frame, validation_frame = validation_frame
  );saveRDS(stacked,paste0(save.dir,'stacked.rds'))

  #-------------------------
  # PERFORAMNCE COMPARISONS
  #-------------------------
  all <- c(learners,stacked)

  # PERFROMANCE ON TRAINING DATA
  train.logloss <- sapply(all, h2o.logloss) # for logloss, the lower, the better
  train.auc     <- sapply(all, h2o.auc)     # For AUC, the bigger, the better
  # PERFROMANCE ON CROSS-VALIDATION DATA
  perf <- lapply(all, h2o.performance, testing_frame)
  cv.logloss <- sapply(perf, h2o.logloss)
  cv.auc     <- sapply(perf, h2o.auc)
  # A CONSOLIDATED VIEW
  all.perf <- noquote(cbind( c('rf','gbm','glm','dl','stacked')
         ,train.logloss=train.logloss
         ,cv.logloss=cv.logloss
         ,train.auc=train.auc
         ,cv.auc=cv.auc
         ));saveRDS(all.perf,paste0(save.dir,'all.perf.rds')
          );all.perf
}

if (skip <- TRUE){
perf[[5]]@metrics$cm
perf[[5]]@metrics$AUC
perf[[5]]@metrics$thresholds_and_metric_scores[c('tpr','fpr')]
}

#------------
# ROC CURVES
#------------
if(skip <- TRUE){

  plot(rf@model$cross_validation_metrics,type='roc'
       ,col='blue',typ='l',lwd=2)
  par(new = TRUE)
  plot(gbm@model$cross_validation_metrics
       ,type = "roc", col = "red", typ = "l")
  par(new = TRUE)
  plot(glm@model$cross_validation_metrics
       ,type = "roc", col = "green", typ = "l")
  par(new = TRUE)
  plot(dl@model$cross_validation_metrics
       ,type = "roc", col = "orange", typ = "l")
  par(new = TRUE)
  plot(stacked.perf.test@metrics$thresholds_and_metric_scores[c('fpr','tpr')]
       ,type = "roc", col = "black", typ = "l")

  #----------
  mapply( function(x){
    plot(paste0(x,'@model$cross_validation_metrics'),type='roc'
       ,col='blue',typ='l',lwd=2)}
    ,list(
      stacked.perf.test@metrics$thresholds_and_metric_scores[c('tpr','fpr')]
      ,rf.perf.test@metrics$thresholds_and_metric_scores[c('tpr','fpr')]
      ,gbm.perf.test@metrics$thresholds_and_metric_scores[c('tpr','fpr')]
      ,glm.perf.test@metrics$thresholds_and_metric_scores[c('tpr','fpr')]
      ,dl.perf.test@metrics$thresholds_and_metric_scores[c('tpr','fpr')]
    )
  )

}

library(tidyverse)
roc.plotly <- ggplotly(

  roc.gg <- list(rf,gbm,glm,dl,stacked) %>%
    # map a function to each element in the list
    map(function(x) x %>% h2o.performance(valid=T) %>%
        #      map(function(x) x %>% h2o.ensemble_performance(newdata=validation_frame) %>%
        # from all these 'paths' in the object
        .@metrics %>% .$thresholds_and_metric_scores %>%
        # extracting true positive rate and false positive rate
        .[c('tpr','fpr')] %>%
        # add (0,0) and (1,1) for the start and end point of ROC curve
        add_row(tpr=0,fpr=0,.before=T) %>%
        add_row(tpr=0,fpr=0,.before=F)) %>%
    # add a column of model name for future grouping in ggplot2
    map2(c('random forest','gbm','glm','deep learning','stacked ensemble'),
       function(x,y) x %>% add_column(model=y)) %>%
    # reduce multiple data.frames to one
    reduce(rbind) %>%
    # plot fpr and tpr, map model to color as grouping
    ggplot(aes(fpr,tpr,col=model))+
    geom_line()+
    geom_segment(aes(x=0,y=0,xend=1, yend=1),linetype=2,col='grey')+
    xlab('False Positive Rate')+
    ylab('True Positive Rate')+
    ggtitle('ROC Curves of Ensemble Learning')

);saveRDS(roc.plotly,paste0(save.dir,'roc.plotly.rds')
);roc.plotly

stacked.pred <- predict(stacked,newdata=testing_frame)

if(skip){
if (!require('OptimalCutpoints')) install.packages('OptimalCutpoints'); library(OptimalCutpoints)

optimal_cutpoint<-optimal.cutpoints( X=stacked ,status='1'
  ,tag.healthy = 1 ,methods=c('Youden','MaxSpSe','SpEqualSe')
  ,data=testing_frame ,pop.prev=NULL ,categorical.cov='readmitted'
  ,control=control.cutpoints(), ci.fit=FALSE, conf.level=0.95, trace=FALSE)
}

#----------------------------------------
# Visualizing error types of predictions
#----------------------------------------
#cat('Classification threshold:', threshold)

stacked.pred.test.df <- (as.data.frame(stacked.pred$predict))
y.test.df <- as.data.frame(testing_frame$readmitted)

pred_type <- function(pred, obs, cutoff=threshold) {
  ptype <- rep(NA,nrow(obs))
  ptype <-
    ifelse(pred >= cutoff & obs == 1, "TP",
     ifelse(pred >= cutoff & obs == 0, "FP",
      ifelse(pred < cutoff & obs == 1, "FN",
       ifelse(pred < cutoff & obs == 0, "TN", '???'))))
  return (ptype)
}

pred.type <- cbind(

   stacked.pred.test.df
  ,pred_type(stacked.pred.test.df ,y.test.df ,threshold)
  ,ifelse(stacked.pred.test.df < threshold ,'not readmitted','readmitted')
  ,y.test.df
  ,ifelse(y.test.df==0 ,'not readmitted','readmitted')

);colnames(pred.type) <- c('pred','type' ,'pred.class','obs','obs.class')

#----------------------------------------------------
# 2D scatter Plot of Prediciton by Ensemble Learning
#----------------------------------------------------
if (!require('plotly')) install.packages('plotly');library(plotly)
if (!require('RColorBrewer')) install.packages('RColorBrewer');library(RColorBrewer)

pred.type <- pred.type[order(pred.type$obs, decreasing = FALSE),]
xmark <- nrow(pred.type[pred.type$obs==0,])
xmax <- nrow(pred.type)

(p2d.class <- plot_ly( pred.type ,x = ~1:xmax ,y = ~pred.type[,'obs']
                       ,name='observation' ,type='scatter'
   # OPTIONAL BLEOW --------------
   ,color = ~pred.type[,'obs']
   ,colors=c('blue','red')  # 0 ,1
   # -----------------------------
   #  ,width=1500 ,height=700 #,margin=5
   ,hoverinfo = 'text' ,text = ~paste(
     'test data:' ,pred.type[,'obs']
    ,'\nclassification:' ,pred.type[,'obs.class'])
   ,mode = 'markers+lines'
   ,marker = list( size = 5 ,opacity = 0.5
   #,color = pred.type ,colorbar=list(title = "Viridis")
   #,color = colorRampPalette(brewer.pal(12,'Set1'))(2000)
   ,line = list( color='black' ,width=1))
) %>% add_trace(y = ~pred.type[,'pred'] ,name='prediction' ,mode = 'markers'
        ,hoverinfo = 'text' ,text = ~paste(
           'prediction:' ,round(pred.type[,'pred'],4)
          ,'\nthreshold:' ,threshold
          ,'\nclassification:' ,pred.type[,'pred.class']
          ,'\n-------------------------------------------'
          ,'\ntest data:' ,pred.type[,'obs']
          ,'\nclassification:' ,pred.type[,'obs.class']
          ,'\n-------------------------------------------'
          ,'\ntype:' ,pred.type[,'type'])
) %>% layout( title=sprintf(
  'Confusion Matrix of Ensemble Learning with Test Data\n(%i Obs.)', nrow(y.test.df))
  ,xaxis=list(title='observation')
  ,yaxis=list(title='prediction')
  #,plot_bgcolor=info['backgroundcolor']
  ,annotations=list( text=sprintf('<b>%s</b>',c('FP','TN','TP','FN'))
    ,x=c(xmark*0.5 ,xmark*0.5 ,xmark+(xmax-xmark)*0.5 ,xmark+(xmax-xmark)*0.5)
    ,y=c( threshold+(1-threshold)*0.5,threshold*0.5
         ,threshold+(1-threshold)*0.5,threshold*0.5)
    ,font=list(size=30),showarrow=FALSE)
  ,shapes = list(
#    list( type = "line",line = list( color = 'black' ,width = 1, dash = 'dot'),
#          x0 = 0, x1 = nrow(pred.type), y0 = threshold, y1 = threshold )
   list( type = "line",line = list( color = 'black' ,width = 1.3),
          x0 = xmark, x1 = xmark, y0 = 0, y1 = 1 )
    )
) %>% add_trace( name='threshold'  #,showlegend=FALSE
                 ,y= threshold
                 ,line = list( color = 'black' ,width = 1, dash = 'dot')
                 ,marker = list( size = 1,color = 'balck')
                 ,hoverinfo = 'text' ,text=sprintf('threshold = %.2f', threshold)
));saveRDS(p2d.class, paste0(save.dir,'p2d.class.rds'
));htmlwidgets::saveWidget(p2d.class, paste0(prefix,'.p2d.class.html')
);file.move(paste0(prefix,'.p2d.class.html'), output.folder)

