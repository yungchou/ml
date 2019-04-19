{
  #-----------------
  # CUSTOM SETTINGS
  #-----------------
  set.seed(0-0)  # DEFAULT SEED

  info <- c()
  info['percent'] <- percent <- 5  # obs. sampled from imported dataset
  info['output.folder']  <- output.folder  <- 'result'
  info['data.partition.train'] <- data.partition.train <- 0.6
  info['data.partition.test']  <- data.partition.test  <- 0.2
  info['data.partition.hold']  <- data.partition.hold  <- 0.2
  info['use.train.balanced'] <- use.train.balanced <- FALSE  # FALSE uses smote
  info['cvControl'] <- cvControl <- 4
  info['run.all']   <- run.all   <- FALSE  # run all SL algorithms
  info['threshold'] <- threshold <- 0.5    # threshold for classfication
  info['backgroundcolor'] <- backgroundcolor <- 'rgb(250,250,250)'
  info['gridcolor']       <- gridcolor       <- 'rgb(170,170,170)'

  saveRDS(info,'info.rds')
}

#-----------
# LIBRARIES
#-----------
{
if (!require('filesstrings')) install.packages('filesstrings'); library(filesstrings)
if (!require('dplyr')) install.packages('dplyr'); library(dplyr)
if (!require('plotly')) install.packages('plotly'); library(plotly)

if (!require('munsell')) install.packages("munsell"); library(munsell)
if (!require('ggplot2')) install.packages('ggplot2'); library(ggplot2)

if (!require('SuperLearner')) install.packages('SuperLearner'); library(SuperLearner)
if (!require('ranger' )) install.packages('ranger' ); library(ranger )
if (!require('xgboost')) install.packages('xgboost'); library(xgboost)
if (!require('caret'  )) install.packages('caret'  ); library(caret  )
}

if(run.all){  # OPTIONAL

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

#if (!require('h2o')) install.packages('h2o'); library(h2o)

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

#-------------------
# DATA PARTITIONING
#-------------------

# Imported data
imported.file <- read.csv( (info['imported.file'] <- 'data/capstone.dataimp.csv') )

df.imported <- imported.file[-1] # data set with Boruta selected fetures

info['total.obs,imported'] <- total.obs.imported <- nrow(df.imported)

# Employed data
df <- df.imported[1:(info['employed.obs'] <- floor(total.obs.imported*percent/100)),]

part <- sample(3 ,nrow(df) ,replace=TRUE ,prob=c(
  data.partition.train ,data.partition.test ,data.partition.hold
))

# Setting up 2 training sets
train.org   <- df[part==1,]  # original training data
train.smote <- df[part==1,]  # SMOTEd training data

test <- df[part==2,]
hold <- df[part==3,]  # for cross validation

# For RMD use
info['nrow(train.org)'] <- nrow(train.org)
info['nrow(test)']      <- nrow(test)
info['nrow(hold)']      <- nrow(hold)
saveRDS(info,'info.rds')

#-----------------
# CLASS IMBALANCE
#-----------------

if(use.train.balanced) {
  # METHOD 1 - USING EUQAL NUMBER OF 0's and 1's
  print('Customizing and Rebalancing the label data')

  ones <- train.org[train.org$readmitted==1,]
  zeros <- (train.org[train.org$readmitted==0,])[1:nrow(ones),]
  train.balanced <- rbind(zeros,ones)

  # The label data distribution originally
  train.org$readmitted <- as.factor(train.org$readmitted
  );saveRDS(train.org$readmitted,'train.org.readmitted.rds')

  # the label data with balanced distribution
  train.balanced$readmitted <- as.factor(train.balanced$readmitted
  );saveRDS(train.balanced$readmitted,'train.balanced.readmitted.rds')

}else{

  # METHOD 2 - USING SMOTE
  # Using SMOTE to fix class imbalance
  print('Using SMOTE to rebalance the label data')

  # The label data distribution originally
  train.org$readmitted <- as.factor(train.org$readmitted
  );saveRDS(train.org$readmitted,'train.org.readmitted.rds')

  if (!require('DMwR')) install.packages('DMwR'); library(DMwR)
  # the label data with SMOTE'd distribution
  train.smote$readmitted <- as.factor(train.smote$readmitted
  );train.smote <- SMOTE(readmitted~., train.smote,
                         perc.over=290, perc.under=140
  );saveRDS(train.smote$readmitted,'train.balanced.readmitted.rds')

  train.balanced <- train.smote

}

table(train.org$readmitted)
table(train.balanced$readmitted)

par(mfrow=c(1,2)
);plot(train.org['readmitted'] ,las=1 ,col='lightblue' ,xlab='label(readmitted)'
       ,main=sprintf('Class Imbalance of Orginal Training Label\n(%i vb. %i)'
         ,table(train.org$readmitted)[[1]],table(train.org$readmitted)[[2]])
);plot(train.balanced['readmitted'] ,las=1 ,col='lightgreen' ,xlab='label(readmitted)'
       ,main=sprintf('Customized/Rebalanced Training Label\n(%i vs. %i)'
         ,table(train.balanced$readmitted)[[1]],table(train.balanced$readmitted)[[2]])
);par(mfrow=c(1,1))

# Now converting the label data back for later model fitting
# When converted from factor to numberic,
# '0' and '1' become '1' and '2'. And must change it back to '0' and '1'
train.org$readmitted      <- as.numeric(factor(train.org$readmitted))-1
train.balanced$readmitted <- as.numeric(factor(train.balanced$readmitted))-1

train <- train.balanced

#--------------
# HOUSEKEEPING
#--------------
# Setting up file structure
  info['prefix'] <- prefix <- paste0('SL' ,train.org.obs <- nrow(train.org)
);info['save.dir'] <- save.dir <- paste0(info['output.folder'],'/',prefix,'.'
);info['train.balanced.obs'] <- train.balanced.obs <- nrow(train.balanced
);info['train.obs'] <- train.obs <- nrow(train

);saveRDS(info,'info.rds')

#---------------------
# SEPARAING THE LABEL
#---------------------
{

  # Sort the label of tringin data in ascending order,
  # i.e. zeros than ones, for better viewing
  train <- train[order(train$readmitted),]
  test  <- test [order(test$readmitted) ,]
  hold  <- hold [order(hold$readmitted) ,]

  x.train <- train[,!(colnames(df) %in% c('readmitted'))]
  y.train <- train[,'readmitted']

  x.test  <- test[,!(colnames(df) %in% c('readmitted'))]
  y.test  <- test[,'readmitted']

  x.hold  <- hold[,!(colnames(df) %in% c('readmitted'))]
  y.hold  <- hold[,'readmitted']

  saveRDS(x.train,paste0(save.dir,'x.train.rds'))
  saveRDS(y.train,paste0(save.dir,'y.train.rds'))

  saveRDS(x.test ,paste0(save.dir,'x.test.rds' ))
  saveRDS(y.test ,paste0(save.dir,'y.test.rds' ))

  saveRDS(x.hold ,paste0(save.dir,'x.hold.rds' ))
  saveRDS(y.hold ,paste0(save.dir,'y.hold.rds' ))

}
#--------------
# HYPERAMETERS
#--------------
# BASELINE
xgboost.custom <- create.Learner('SL.xgboost'
  ,tune=list( # default: ntree=500, depth=4, shrinkage=0.1, minobs=10
    ntrees=c(500,1000) ,max_depth=4
   ,shrinkage=c(0.001,0.01,0.1) ,minobspernode=c(10)
   )
  ,detailed_names = TRUE ,name_prefix = 'xgboost'
);saveRDS(xgboost.custom,paste0(save.dir,'xgboost.custom.rds')

);ranger.custom <- create.Learner('SL.ranger'
 ,tune = list( # default: num.trees=1000, mtry=sqrt(variables) or var/3
    num.trees = c(1000,2000)
   ,mtry = floor(sqrt(ncol(x.train))*c(1,2))
   )
 ,detailed_names = TRUE ,name_prefix = 'ranger'
);saveRDS(ranger.custom,paste0(save.dir,'ranger.custom.rds'))

if(run.below <- FALSE){
  glmnet.custom <-  create.Learner('SL.glmnet'
   ,tune = list(
     alpha  = seq(0 ,1 ,length.out=10)  # (0,1)=>(ridge, lasso)
    ,nlambda = seq(0 ,10 ,length.out=10)
     )
   ,detailed_names = TRUE ,name_prefix = 'glmnet'
  );saveRDS(glmnet.custom,paste0(save.dir,'glmnet.custom.rds'))
}

#ranger.custom <- function(...) SL.ranger(...,num.trees=1000, mtry=5)
#kernelKnn.custom <- function(...) SL.kernelKnn(...,transf_categ_cols=TRUE)

#-----------------------
# SUPERLEARNER SETTINGS
#-----------------------
family   <- 'binomial'    #'gaussian'
nnls     <- 'method.NNLS' # NNLS-default
auc      <- 'method.AUC'
nnloglik <- 'method.NNloglik'

ifelse(run.all
  ,SL.algorithm <- (listWrappers())[69:110]
  ,SL.algorithm <- c( #'SL.ranger','SL.xgboost','SL.glmnet'
     ranger.custom$names
    ,xgboost.custom$names
    #,glmnet.custom$names
    )
);saveRDS(SL.algorithm,paste0(save.dir,'SL.algorithm.rds'))

#-------------------------------
# MULTICORE/PARALLEL PROCESSING
#-------------------------------
if (!require('parallel')) install.packages('parallel'); library(parallel)

cl <- makeCluster(detectCores()-1)

clusterExport(cl, c( listWrappers()

  ,'SuperLearner','CV.SuperLearner','predict.SuperLearner','cvControl'
  ,'x.train','y.train','x.test','y.test','x.hold','y.hold'
  ,'family','nnls','auc','nnloglik' ,'save.dir'

  ,'SL.algorithm'
  ,ranger.custom$names ,xgboost.custom$names #,glmnet.custom$names

  ))

clusterSetRNGStream(cl, iseed=135)

# Load libraries on workers
clusterEvalQ(cl, {
  library(SuperLearner);library(caret)
  library(ranger);library(xgboost)
#library(e1071) #;library(nnet) #;library(MASS)
#library(glmnet) #;library(randomForest)
#library(kernlab) #;ibrary(arm) #;library(klaR)

})

#----------------------------------
# FITTING MODEL WITH TRAINING DATA
#----------------------------------
clusterEvalQ(cl, {

  ensem.auc <- SuperLearner(Y=y.train ,X=x.train #,verbose=TRUE
    ,family=family,method=auc,SL.library=SL.algorithm,cvControl=list(V=cvControl)
    );saveRDS(ensem.auc ,paste0(save.dir,'ensem.auc.rds'))

  ensem.nnls <- SuperLearner(Y=y.train ,X=x.train #,verbose=TRUE
    ,family=family,method=nnls,SL.library=SL.algorithm,cvControl=list(V=cvControl)
    );saveRDS(ensem.nnls ,paste0(save.dir,'ensem.nnls.rds'))

  ensem.nnloglik <- SuperLearner(Y=y.train ,X=x.train #,verbose=TRUE
    ,family=family,method=nnloglik,SL.library=SL.algorithm,cvControl=list(V=cvControl)
    );saveRDS(ensem.nnloglik ,paste0(save.dir,'ensem.nnloglik.rds'))

})

#---------------------------------------
# DOING CROSS VALIDATION WITH HOLD DATA
#---------------------------------------
{
  ensem.cv.nnls.time <- system.time({
    ensem.cv.nnls <- CV.SuperLearner(Y=y.hold ,X=x.hold ,verbose=TRUE
     ,cvControl=list(V=cvControl)#,innerCvControl=list(list(V=cvControl-1))
     ,family=family ,method=nnls ,SL.library=SL.algorithm ,parallel=cl
    );saveRDS(ensem.cv.nnls ,paste0(save.dir,'ensem.cv.nnls.rds'))
  })

  ensem.cv.auc.time <- system.time({
    ensem.cv.auc <- CV.SuperLearner( Y=y.hold ,X=x.hold #,verbose=TRUE
     ,cvControl=list(V=cvControl)#,innerCvControl=list(list(V=cvControl-1))
     ,family=family ,method=auc ,SL.library=SL.algorithm ,parallel=cl
    );saveRDS(ensem.cv.auc ,paste0(save.dir,'ensem.cv.auc.rds'))
  })

  ensem.cv.nnloglik.time <- system.time({
    ensem.cv.nnloglik <- CV.SuperLearner( Y=y.hold ,X=x.hold #,verbose=TRUE
     ,cvControl=list(V=cvControl)#,innerCvControl=list(list(V=cvControl-1))
     ,family=family ,method=nnloglik ,SL.library=SL.algorithm ,parallel=cl
    );saveRDS(ensem.cv.nnloglik ,paste0(save.dir,'ensem.cv.nnloglik.rds'))
  })

  ensem.cv.time <- t(
    cbind(ensem.cv.nnls.time,ensem.cv.auc.time,ensem.cv.nnloglik.time)
  )[,c('user.self', 'sys.self', 'elapsed')
  ];saveRDS(ensem.cv.time,paste0(save.dir,'ensem.cv.time'))
}

stopCluster(cl)

#--------------------------
# READING IN SAVED RESULTS
#--------------------------
{ # For reproducing ensemble learning output, starts here.

  # Reading in info and repopulating referenced constants
  info <- readRDS('info.rds');save.dir <- info['save.dir']
  train.obs <- as.integer(info['train.obs'])
  backgroundcolor <- info['backgroundcolor']

  ensem.nnls     <- readRDS(paste0(save.dir,'ensem.nnls.rds'))
  ensem.auc      <- readRDS(paste0(save.dir,'ensem.auc.rds' ))
  ensem.nnloglik <- readRDS(paste0(save.dir,'ensem.nnloglik.rds'))

  info['ensem.nnls.times']     <- ensem.nnls$times
  info['ensem.auc.times']      <- ensem.auc$times
  info['ensem.nnloglik.times'] <- ensem.nnloglik$times

  saveRDS(info, 'info.rds')

  ensem.cv.nnls     <- readRDS(paste0(save.dir,'ensem.cv.nnls.rds'))
  ensem.cv.auc      <- readRDS(paste0(save.dir,'ensem.cv.auc.rds'))
  ensem.cv.nnloglik <- readRDS(paste0(save.dir,'ensem.cv.nnloglik.rds'))
}

#-------------------------------------------
# ASSESSING THE RISKS OF EVALUATION METHODS
#-------------------------------------------
compare <- noquote(cbind(
   ensem.nnls$cvRisk,ensem.nnls$coef
  ,ensem.auc$cvRisk,ensem.auc$coef
  ,ensem.nnloglik$cvRisk,ensem.nnloglik$coef
));colnames(compare) <- c(
   'nnls.cvRisk','nnls.coef'
  ,'auc.cvRisk','auc.coef'
  ,'nnloglik.cvRisk','nnloglik.coef'
);saveRDS(compare,paste0(save.dir,'compare.rds'))

#compare

#---------------------------------------
# 2D Scatter Plot - Risks & Coefficient
#---------------------------------------
(p2d.risk_coef <- plot_ly( as.data.frame(compare)

  ,x=~ensem.nnls$libraryNames
  ,y=~ensem.nnls$cvRisk, name='risk nnls'
  ,hoverinfo = 'text' ,text = ~paste(
    'function:' ,ensem.nnls$libraryNames
    ,'\nrisk nnls:' ,ensem.nnls$cvRisk)

  ,type='scatter',mode = 'markers'
#  ,width=1500 ,height=750 #,margin=5
  ,marker=list( size = 7, opacity = 0.5
    ,line=list( color='black' ,width=1
     #,shape='spline' ,smoothing=1.3
      ))
) %>% add_trace(y = ~ensem.auc$cvRisk, name='risk auc'
                ,hoverinfo = 'text' ,text = ~paste(
                  'function:' ,ensem.auc$libraryNames
                  ,'\nrisk auc:' ,ensem.auc$cvRisk)
) %>% add_trace(y = ~ensem.nnloglik$cvRisk, name='risk nnloglik'
                ,hoverinfo = 'text' ,text = ~paste(
                  'function:' ,ensem.nnloglik$libraryNames
                  ,'\nrisk nnloglik:' ,ensem.nnloglik$cvRisk)

) %>% add_trace(y = ~ensem.nnls$coef, name='coef nnls',yaxis='y2'
                ,hoverinfo = 'text' ,text = ~paste(
                  'function:' ,ensem.nnls$libraryNames
                 ,'\ncoef nnls:' ,ensem.nnls$coef)
) %>% add_trace(y = ~ensem.auc$coef, name='coef auc',yaxis='y2'
                ,hoverinfo = 'text' ,text = ~paste(
                  'function:' ,ensem.auc$libraryNames
                  ,'\ncoef auc:' ,ensem.auc$coef)
) %>% add_trace(y = ~ensem.nnloglik$coef, name='coef nnloglik',yaxis='y2'
                ,hoverinfo = 'text' ,text = ~paste(
                  'function:' ,ensem.nnloglik$libraryNames
                  ,'\ncoef nnloglik:' ,ensem.nnloglik$coef)

) %>% layout( title=sprintf("Risk and Coefficient of Learners Based on Method \n(SMOTE'd Training Data = %i obs.)", train.obs)
          ,xaxis=list(title='')
          ,yaxis=list(title='risk'
            #,range=c(min(ensem.nnls$cvRisk),max(ensem.nnls$cvRisk)+0.1)
          )
          ,yaxis2=list(title='coefficient' #
            ,range=c(min(ensem.nnls$coef,ensem.auc$coef,ensem.nnloglik$coef)-0.1
                    ,max(ensem.nnls$coef,ensem.auc$coef,ensem.nnloglik$coef)+0.1)
                    ,overlaying='y' ,side='right')
          ,margin=list() #l=50, r=50, b=50, t=50, pad=4
          ,plot_bgcolor=backgroundcolor
#) %>% rangeslider(
));saveRDS(p2d.risk_coef, paste0(save.dir,'p2d.risk_coef.rds'
));htmlwidgets::saveWidget(
  p2d.risk_coef,paste0(prefix,'.p2d.risk_coef.html')
);file.move(paste0(prefix,'.p2d.risk_coef.html'), output.folder)

#%>% add_lines(y = ~ensem.auc$cvRisk, colors = "black", alpha = 0.2)

#-------------------
# MAKING PREDICTION
#-------------------
{
  pred.nnls <- predict.SuperLearner(ensem.nnls ,x.test ,onlySL=TRUE)
  saveRDS(pred.nnls,paste0(save.dir,'pred.nnls.rds'))

  pred.auc <- predict.SuperLearner(ensem.auc ,x.test ,onlySL=TRUE)
  saveRDS(pred.auc,paste0(save.dir,'pred.auc.rds'))

  pred.nnloglik <- predict.SuperLearner(ensem.nnloglik ,x.test ,onlySL=TRUE)
  saveRDS(pred.nnloglik,paste0(save.dir,'pred.nnloglik.rds'))
}
#----------------------------------
# Summary of Predictions by Method
#----------------------------------
pred.summary <- noquote(cbind(
  summary(pred.nnls$pred)
 ,summary(pred.auc$pred)
 ,summary(pred.nnloglik$pred))
);colnames(pred.summary) <- c('nnls','auc','nnloglik'
);saveRDS(pred.summary,paste0(save.dir,'pred.summary.rds')
);pred.summary

#----------------------------------------
# Visualizing error types of predictions
#----------------------------------------
#cat('Classification threshold:', threshold)

pred_type <- function(plist, label=y.test, cutoff=0.5) {
  ptype <- rep(NA, length(y.test))
  ptype <-
    ifelse(plist >= cutoff & label == 1, "TP",
      ifelse(plist >= cutoff & label == 0, "FP",
        ifelse(plist < cutoff & label == 1, "FN",
          ifelse(plist < cutoff & label == 0, "TN", '???'))))
  return (ptype)
}

pred.type <- noquote(cbind(

   pred.nnls$pred ,pred_type(pred.nnls$pred ,y.test ,threshold)
  ,ifelse(pred.nnls$pred < threshold ,'not readmitted','readmitted')

  ,pred.auc$pred ,pred_type(pred.auc$pred ,y.test ,threshold)
  ,ifelse(pred.auc$pred < threshold ,'not readmitted','readmitted')

  ,pred.nnloglik$pred ,pred_type(pred.nnloglik$pred ,y.test ,threshold)
  ,ifelse(pred.nnloglik$pred < threshold ,'not readmitted','readmitted')

  ,y.test ,ifelse(y.test==0 ,'not readmitted','readmitted')

));colnames(pred.type) <- c(
   'nnls'     ,'nnls.type'     ,'nnls.prediction'
  ,'auc'      ,'auc.type'      ,'auc.prediction'
  ,'nnloglik' ,'nnloglik.type' ,'nnloglik.prediction'
  ,'label'    ,'description'
);saveRDS(pred.type,paste0(save.dir,'pred.type.rds'))

#pred.type

#----------------------------------------------------
# 2D scatter Plot of Prediciton by Ensemble Learning
#----------------------------------------------------
pred.type <- readRDS(paste0(save.dir,'pred.type.rds'))

if (!require('plotly')) install.packages('plotly');library(plotly)
if (!require('RColorBrewer')) install.packages('RColorBrewer');library(RColorBrewer)

#cat('Classification threshold:', threshold)

(p2d <- plot_ly( as.data.frame(pred.type)
  ,x = ~1:nrow(pred.type)
  ,y = ~pred.type[,'label'] ,name='label'
# OPTIONAL BLEOW -----------------------------
#,color = ~pred.type[,'description']
#,colors=c('red','blue')
# TOPIONAL ABOVE -----------------------------
  ,hoverinfo = 'text' ,text = ~paste(
      'label:' ,pred.type[,'label']
     ,'\nobservation:' ,pred.type[,'description'])
  ,type='scatter'
#  ,width=1500 ,height=700 #,margin=5
  ,mode = 'markers+lines'
  ,marker = list( size = 7 ,opacity = 0.5
     #,color = pred.type ,colorbar=list(title = "Viridis")
     #,color = colorRampPalette(brewer.pal(12,'Set1'))(2000)
     ,line = list( color='black' ,width=1))
) %>% add_trace(y = ~pred.type[,'nnls'] ,name='nnls' ,mode = 'markers'
                ,hoverinfo = 'text' ,text = ~paste(
                   'nnls:' ,pred.type[,'nnls']
                  ,'\nthreshold:' ,threshold
                  ,'\nprediction:' ,pred.type[,'nnls.prediction']
                  ,'\n-------------------------------------------'
                  ,'\nlabel:' ,pred.type[,'label']
                  ,'\nobservation:' ,pred.type[,'description']
                  ,'\n-------------------------------------------'
                  ,'\ntype:' ,pred.type[,'nnls.type'])
) %>% add_trace(y = ~pred.type[,'auc'] ,name='auc' ,mode = 'markers'
                ,hoverinfo = 'text', text = ~paste(
                   'auc:' ,pred.type[,'auc']
                  ,'\nthreshold:' ,threshold
                  ,'\nprediction:' ,pred.type[,'auc.prediction']
                  ,'\n-------------------------------------------'
                  ,'\nlabel:' ,pred.type[,'label']
                  ,'\nobservation:' ,pred.type[,'description']
                  ,'\n-------------------------------------------'
                  ,'\ntype:' ,pred.type[,'auc.type'])
) %>% add_trace(y = ~pred.type[,'nnloglik'] ,name='nnloglik' ,mode = 'markers'
                ,hoverinfo = 'text', text = ~paste(
                   'nnloglik:' ,pred.type[,'nnloglik']
                  ,'\nthreshold:' ,threshold
                  ,'\nprediction:' ,pred.type[,'nnloglik.prediction']
                  ,'\n-------------------------------------------'
                  ,'\nlabel:' ,pred.type[,'label']
                  ,'\nobservation:' ,pred.type[,'description']
                  ,'\n-------------------------------------------'
                  ,'\ntype:' ,pred.type[,'nnloglik.type'])
) %>% layout( title=sprintf(
  'Predicitons by Ensemble Learning with Test Data\n(Threshold = %.2f and %i Obs.)', threshold, nrow(x.test))
  ,xaxis=list(title='observation')
  ,yaxis=list(title='prediction')
  ,plot_bgcolor=info['backgroundcolor']
  ,annotations=list( text=''  # legend title
  ,yref='paper',xref='paper'
  ,y=1.025 ,x=1.09 ,showarrow=FALSE)
) %>% add_trace( name='threshold'  #,showlegend=FALSE
  ,y= threshold
  ,line = list( color = 'black' ,width = 1, dash = 'dot')
  ,marker = list( size = 1,color = 'balck'
#    ,line = list( color = 'black' ,width = 1, dash = 'dot')
)
  ,hoverinfo = 'text' ,text=sprintf('threshold = %.2f', threshold)
#) %>% rangeslider(
));saveRDS(p2d, paste0(save.dir,'p2d.rds'
));htmlwidgets::saveWidget(p2d,paste0(prefix,'.p2d.html')
);file.move(paste0(prefix,'.p2d.html'), output.folder)

#----------------------------------------
# 2D SCATTER PLOT SHOWING CLASSIFICAITON
#----------------------------------------
(p2d.class <- plot_ly( as.data.frame(pred.type)
  ,x = ~1:nrow(pred.type)
  ,y = ~pred.type[,'label'] #,name='label'

  ,color = ~pred.type[,'description']
  ,colors=c('blue','red')  # 0 ,1

  ,hoverinfo = 'text' ,text = ~paste(
    'label:' ,pred.type[,'label']
   ,'\nobservation:' ,pred.type[,'description'])
  ,type='scatter'
#  ,width=1500 ,height=700 #,margin=5

  ,mode = 'markers+lines'
  ,marker = list( size = 7 ,opacity = 0.8
   #,color = pred.type ,colorbar=list(title = "Viridis")
   #,color = colorRampPalette(brewer.pal(12,'Set1'))(2000)
   #,line = list( color = 'black' ,width = 1)
   )
) %>% add_trace(y = ~pred.type[,'nnls'] ,name='nnls' ,mode = 'markers'
                ,marker = list( size = 7 ,opacity = 0.6
                                ,line = list( color = 'black' ,width = 1))
                ,hoverinfo = 'text' ,text = ~paste(
                  'nnls:' ,pred.type[,'nnls']
                  ,'\nthreshold:' ,threshold
                  ,'\nprediction:' ,pred.type[,'nnls.prediction']
                  ,'\n-------------------------------------------'
                  ,'\nlabel:' ,pred.type[,'label']
                  ,'\nobservation:' ,pred.type[,'description']
                  ,'\n-------------------------------------------'
                  ,'\ntype:' ,pred.type[,'nnls.type'])
) %>% add_trace(y = ~pred.type[,'auc'] ,name='auc' ,mode = 'markers'
                ,marker = list( size = 7 ,opacity = 0.4
                  ,line = list( color = 'black' ,width = 1))
                ,hoverinfo = 'text', text = ~paste(
                  'auc:' ,pred.type[,'auc']
                  ,'\nthreshold:' ,threshold
                  ,'\nprediction:' ,pred.type[,'auc.prediction']
                  ,'\n-------------------------------------------'
                  ,'\nlabel:' ,pred.type[,'label']
                  ,'\nobservation:' ,pred.type[,'description']
                  ,'\n-------------------------------------------'
                  ,'\ntype:' ,pred.type[,'auc.type'])
) %>% add_trace(y = ~pred.type[,'nnloglik'] ,name='nnloglik' ,mode = 'markers'
                ,marker = list( size = 7 ,opacity = 0.2
                  ,line = list( color = 'black' ,width = 1))
                ,hoverinfo = 'text', text = ~paste(
                  'nnloglik:' ,pred.type[,'nnloglik']
                  ,'\nthreshold:' ,threshold
                  ,'\nprediction:' ,pred.type[,'nnloglik.prediction']
                  ,'\n-------------------------------------------'
                  ,'\nlabel:' ,pred.type[,'label']
                  ,'\nobservation:' ,pred.type[,'description']
                  ,'\n-------------------------------------------'
                  ,'\ntype:' ,pred.type[,'nnloglik.type'])
) %>% layout( title=sprintf(
  'Predicitons by Ensemble Learning with Test Data\n(Threshold = %.2f and %i Obs.)', threshold, nrow(x.test))
  ,xaxis=list(title='observation')
  ,yaxis=list(title='prediction')
  ,plot_bgcolor=backgroundcolor
  ,annotations=list( text=''  # legend title
                     ,yref='paper',xref='paper'
                     ,y=1.025 ,x=1.09 ,showarrow=FALSE)
) %>% add_trace( name='threshold'  #,showlegend=FALSE
                 ,y= threshold
                 ,line = list(color='black',width=1,dash='dot')
                 ,marker = list(size=1,color='balck'
                   #    ,line=list(color='black',width=1,dash='dot')
                 )
                 ,hoverinfo = 'text' ,text=sprintf('threshold = %.2f', threshold)
#) %>% rangeslider(
));saveRDS(p2d.class, paste0(save.dir,'p2d.class.rds'
));htmlwidgets::saveWidget(p2d.class,paste0(prefix,'.p2d.class.html')
);file.move(paste0(prefix,'.p2d.class.html'), output.folder)

#-----------------
# 3D Scatter Plot
#-----------------
if (!require('plotly')) install.packages('plotly');library(plotly)
if (!require('RColorBrewer')) install.packages('RColorBrewer');library(RColorBrewer)

#cat('Classification threshold:', threshold)

(p3d <- plot_ly( as.data.frame(pred.type)
   ,x = ~pred.type[,'nnls'], y = ~pred.type[,'auc'], z = ~pred.type[,'nnloglik']
   ,hoverinfo = 'text' ,text = ~paste(
      'label:'         ,pred.type[,'label']
     ,'\nobservation:' ,pred.type[,'description']
     ,'\n-------------------------------------------'
     ,'\nthreshold:'   ,threshold
     ,'\n-------------------------------------------'
     ,'\nnnls:'        ,pred.type[,'nnls']
     ,'\nprediction:'  ,pred.type[,'nnls.prediction']
     ,'\ntype:'        ,pred.type[,'nnls.type']
     ,'\n-------------------------------------------'
     ,'\nauc:'        ,pred.type[,'auc']
     ,'\nprediction:' ,pred.type[,'auc.prediction']
     ,'\ntype:'       ,pred.type[,'auc.type']
     ,'\n-------------------------------------------'
     ,'\nnnloglik:'   ,pred.type[,'nnloglik']
     ,'\nprediction:' ,pred.type[,'nnloglik.prediction']
     ,'\ntype:'       ,pred.type[,'nnloglik.type']
      )
,color = ~pred.type[,'description']
,colors=c('blue','red')  # 0 ,1
,marker = list(size = 10 ,opacity = 0.5
# https://moderndata.plot.ly/create-colorful-graphs-in-r-with-rcolorbrewer-and-plotly/
#,color = colorRampPalette(brewer.pal(12,'Set3'))(floor(train.obs/12))
#,colorscale = c(brewer.pal(11,'RdBu')[1],brewer.pal(11,'RdBu')[11]),showscale = TRUE
,line = list( color = 'black' ,width = 0.5))
) %>% add_markers(
) %>% layout( title=sprintf(
  'Predicitons by Ensemble Learning with Test Data\n(Threshold = %.2f and %i Obs.)', threshold, nrow(x.test))
  ,scene = list(
    xaxis = list(title='auc',showbackground=TRUE
                 ,backgroundcolor=backgroundcolor ,gridcolor=gridcolor
                 ,zerolinecolor='rgb(0,0,0)')
   ,yaxis = list(title='nnls',showbackground=TRUE
                 ,backgroundcolor=backgroundcolor,gridcolor=gridcolor
                 ,zerolinecolor='rgb(0,0,0)')
   ,zaxis = list(title='nnloglik',showbackground=TRUE
                 ,backgroundcolor=backgroundcolor,gridcolor=gridcolor
                 ,zerolinecolor='rgb(0,0,0)')
   ,camera = list(
         up=list(x=0, y=0, z=1)
        ,center=list(x=0, y=0, z=0)
        ,eye=list(x=1.75, y=-1.25, z=0.5)  )
)
));saveRDS(p3d, paste0(save.dir,'p3d.rds'
));htmlwidgets::saveWidget(p3d,paste0(prefix,'.p3d.html'
));file.move(paste0(prefix,'.p3d.html'), output.folder)

#-------------------------------------
# CONSUSION MATRIX & ACCURACY PARADOX
#-------------------------------------
# Converting probabilities into classification
pred.type$nnls.converted  <- ifelse(pred.nnls$pred >= threshold,1,0
);pred.type$auc.converted <- ifelse(pred.auc$pred  >= threshold,1,0
);pred.type$nnloglik.converted <- ifelse(pred.nnloglik$pred >= threshold,1,0)

# Confusion Matrix and Accurancy Paradox
cm.nnls <- confusionMatrix(
    factor(pred.type$nnls.converted ), factor(y.test)
);saveRDS(cm.nnls, paste0(save.dir,'cm.nnls.rds')
);cm.auc <- confusionMatrix(
    factor(pred.type$auc.converted), factor(y.test)
);saveRDS(cm.auc, paste0(save.dir,'cm.auc.rds')
);cm.nnloglik <- confusionMatrix(
    factor(pred.type$nnloglik.converted ), factor(y.test)
);saveRDS(cm.nnloglik, paste0(save.dir,'cm.nnloglik.rds')

);mse.nnls <- mean((y.test-pred.nnls$pred)^2
);mse.auc <- mean((y.test-pred.auc$pred)^2
);mse.nnloglik <- mean((y.test-pred.nnloglik$pred)^2

);pred.accuracy.paradox <- noquote(cbind(
   MSE=c(mse.nnls,mse.auc,mse.nnloglik)
  ,Accuracy=c(
     cm.nnls$overall['Accuracy']
    ,cm.auc$overall['Accuracy']
    ,cm.nnloglik$overall['Accuracy']
  )
  ,Sensitity=c(
     cm.nnls$byClass['Sensitivity']
    ,cm.auc$byClass['Sensitivity']
    ,cm.nnloglik$byClass['Sensitivity']
  )
  ,Specificity=c(
    cm.nnls$byClass['Specificity']
    ,cm.auc$byClass['Specificity']
    ,cm.nnloglik$byClass['Specificity']
  )
  ,AllNoRecurr=c(   # NIR
    cm.nnls$overall['AccuracyNull']
    ,cm.auc$overall['AccuracyNull']
    ,cm.nnloglik$overall['AccuracyNull']
  )
  ,AllRecurr=c(
    (cm.nnls$table[3]+cm.nnls$table[4])/
      (cm.nnls$table[1]+cm.nnls$table[2]+cm.nnls$table[3]+cm.nnls$table[4])
    ,(cm.auc$table[3]+cm.auc$table[4])/
      (cm.auc$table[1]+cm.auc$table[2]+cm.auc$table[3]+cm.auc$table[4])
    ,(cm.nnloglik$table[3]+cm.nnloglik$table[4])/
      (cm.nnloglik$table[1]+cm.nnloglik$table[2]+cm.nnloglik$table[3]+cm.nnloglik$table[4])
  ))
);rownames(pred.accuracy.paradox) <- c('nnls','auc','nnloglik'
);saveRDS(pred.accuracy.paradox,paste0(save.dir,'pred.accuracy.paradox.rds')
);pred.accuracy.paradox

#-------------------------
# CROSS VALIDATON OBJECTS
#-------------------------
# Alread read-in earlier
#ensem.cv.nnls     <- readRDS(paste0(save.dir,'ensem.cv.nnls.rds'))
#ensem.cv.auc      <- readRDS(paste0(save.dir,'ensem.cv.auc.rds'))
#ensem.cv.nnloglik <- readRDS(paste0(save.dir,'ensem.cv.nnloglik.rds')
{
  s1 <- summary(ensem.cv.nnls)
  t1 <- table(simplify2array(ensem.cv.nnls$whichDiscreteSL))

  s2 <- summary(ensem.cv.auc)
  t2 <- table(simplify2array(ensem.cv.auc$whichDiscreteSL))

  s3 <- summary(ensem.cv.nnloglik)
  t3 <- table(simplify2array(ensem.cv.nnloglik$whichDiscreteSL))
}

#s1;t1
#s2;t2
#s3;t3

#ensem.cv.nnls$AllSL$'2'$coef
#ensem.cv.nnls$AllSL$'2'$cvRisk

#----------
# Stacking
#----------
ensem.cv.stacking.nnls <- plot(ensem.cv.nnls)+theme_bw();ensem.cv.stacking.nnls
ensem.cv.stacking.auc <- plot(ensem.cv.auc)+theme_bw();ensem.cv.stacking.auc
ensem.cv.stacking.nnloglik <- plot(ensem.cv.nnloglik)+theme_bw();ensem.cv.stacking.nnloglik

#----------------------------------------------------------
# CROSS VALIDATION - ROC CURVE
# (receiver operating characteristic curve)
#
# It plots TPR vs. FPR at different classification thresholds.
# https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc
#----------------------------------------------------------
if (!require('ck37r')) install.packages('ck37r') ;library(ck37r)

(ensem.cv.roc.nnls     <- cvsl_plot_roc(ensem.cv.nnls))
(ensem.cv.roc.auc      <- cvsl_plot_roc(ensem.cv.auc))
(ensem.cv.roc.nnloglik <- cvsl_plot_roc(ensem.cv.nnloglik))

#----------------------------------------------------------
# CROSS VALIDATION - AUC
# (Area under the ROC Curve)
#
# AUC provides an aggregate measure of performance across
# all possible classification thresholds. Consider AUC as
# the probability that the model ranks a random positive
# example more highly than a random negative example.
#
# https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc
#----------------------------------------------------------
if (!require('ck37r')) install.packages('ck37r') ;library(ck37r)

cvsl_weights(ensem.cv.nnls)
cvsl_weights(ensem.cv.auc)
cvsl_weights(ensem.cv.nnloglik)

cvsl_auc(ensem.cv.nnls)
cvsl_auc(ensem.cv.auc)
cvsl_auc(ensem.cv.nnloglik)



