#-----------------
# CUSTOM SETTINGS
#-----------------
percent <- 50    # obs. sampled from imported dataset
class.imbalance <- TRUE

output.folder <- 'result'

data.partition <- c(0.6,0.2,0.2)
cvControl <- 5
run.all   <- FALSE  # run all SL algorithms
threshold <- 0.5    # threshold for classfication

backgroundcolor <- 'rgb(240,240,240)'
gridcolor       <- 'rgb(180,180,180)'

set.seed(0-0)  # PLACE A DEFAULT SEED

#-----------
# LIBRARIES
#-----------
if (!require('filesstrings')) install.packages('filesstrings'); library(filesstrings)
if (!require('dplyr')) install.packages('dplyr'); library(dplyr)
if (!require('plotly')) install.packages('plotly'); library(plotly)

if (!require('munsell')) install.packages("munsell"); library(munsell)
if (!require('ggplot2')) install.packages('ggplot2'); library(ggplot2)

if (!require('SuperLearner')) install.packages('SuperLearner'); library(SuperLearner)
if (!require('ranger' )) install.packages('ranger' ); library(ranger )
if (!require('xgboost')) install.packages('xgboost'); library(xgboost)
if (!require('caret'  )) install.packages('caret', dependencies=c("Depends","Suggests") ); library(caret  )

if(run.all){  # OPTIONAL

if (!require('e1071')) install.packages('e1071'); library(e1071)
if (!require('party')) install.packages('party'); library(party)
if (!require('gam')) install.packages('gam'); library(gam)
if (!require('LogicReg')) install.packages('LogicReg'); library(LogicReg)
if (!require('polspline')) install.packages('polspline'); library(polspline)
if (!require('extraTrees')) install.packages('extraTrees'); library(extraTrees)
if (!require('biglasso')) install.packages('biglasso'); library(biglasso)
if (!require('dbarts')) install.packages('dbarts'); library(dbarts)
if (!require('speedglm')) install.packages('speedglm'); library(speedglm)

if (!requireNamespace("BiocManager", quietly = TRUE)
    )install.packages("BiocManager"
    );BiocManager::install("qvalue", version = "3.8")

#install.packages('installr');library(installr);updateR()

if (!require('sva')) install.packages('sva'); library(sva)
if (!require('SmartSVA')) install.packages('SmartSVA'); library(SmartSVA)

if (!require('mlbench')) install.packages('mlbench'); library(mlbench)
if (!require('rpart')) install.packages('rpart'); library(rpart)

if (!require('h2o')) install.packages('h2o'); library(h2o)

install.packages("devtools");library(devtools
);install_github("AppliedDataSciencePartners/xgboostExplainer",force = TRUE)

if (!require('MASS')) install.packages('MASS'); library(MASS)
if (!require('cvAUC')) install.packages('cvAUC'); library(cvAUC)
if (!require('kernlab')) install.packages('kernlab'); library(kernlab)
if (!require('arm')) install.packages('arm'); library(arm)
if (!require('ipred')) install.packages('ipred'); library(ipred)
if (!require('KernelKnn')) install.packages('KernelKnn'); library(KernelKnn)
if (!require('RcppArmadillo')) install.packages('RcppArmadillo'); library(RcppArmadillo)

}

#------------------
# DATA PREPARATION
#------------------
imported.file <- 'data/capstone.dataimp.csv'
df <- read.csv(imported.file)[-1] # data set with Boruta selected fetures
df <- df[1:(nrow(df)*percent/100),]

if(class.imbalance) {

  df$readmitted <- as.factor(df$readmitted)
  table(df$readmitted)

  if (!require('DMwR')) install.packages('DMwR'); library(DMwR)
  df.smote <- SMOTE(readmitted~., df, perc.over=100, perc.under=220)
  table(df.smote$readmitted)

  prefix <- 's'

  par(mfrow=c(1,2)
  );plot(df['readmitted'],las=1,col='lightblue'
         ,xlab='label(readmitted)',main=sprintf('Imported Data Set\n(%i obs.)',nrow(df))
  );plot(df.smote['readmitted'],las=1,col='lightgreen'
         ,xlab='label(readmitted)',main=sprintf('Employed SMOTEd Data\n(%i obs.)',nrow(df.smote))
  );par(mfrow=c(1,1))

  # When converted from factor to numberic, '0' and '1' become '1' and '2'.
  df.smote$readmitted <- as.numeric(factor(df.smote$readmitted))-1
  #tail(df.smote$readmitted)

  part <- sample(3 ,nrow(df.smote) ,replace=TRUE ,prob=data.partition)
  train <- df.smote[part==1,]
  test  <- df.smote[part==2,]
  hold  <- df.smote[part==3,]  # for cross validation

} else {

  prefix <- 'p'

  part  <- sample(3 ,nrow(df) ,replace=TRUE ,prob=data.partition)
  train <- df[part==1,]
  test  <- df[part==2,]
  hold  <- df[part==3,]  # for cross validation
}

#--------------
# Housekeeping
#--------------
train.obs <- nrow(train)
prefix <- paste0(prefix,train.obs)
save.dir <- paste0(output.folder,'/SL.',prefix,'.')

#output text file name
write(timestamp() ,output.file <- paste0(save.dir,'txt') )
write(paste0('imported file <- ',imported.file) ,output.file ,append=TRUE)

sink(output.file ,append=TRUE
);cat('Total observations = ',train.obs);data.usage <- noquote(
    cbind(c('train','test','hold'),c(nrow(train),nrow(test),nrow(hold)))
);colnames(data.usage) <- c('usage','obs.');data.usage;sink()

# Check the index of 'readmitted'
x.train <- train[,-27]
y.train <- train[, 27]

x.test  <- test[,-27]
y.test  <- test[, 27]

x.hold  <- hold[,-27]
y.hold  <- hold[, 27]

#---------
# Tuning
#---------
xgboost.custom <- create.Learner('SL.xgboost'
  ,tune=list(
    ntrees=c(500,1000) ,max_depth=4   # 1:4
   ,shrinkage=c(0.01,0.1) ,minobspernode=c(10,30)
   )
  ,detailed_names = TRUE ,name_prefix = 'xgboost'
)

ranger.custom <- create.Learner('SL.ranger'
 ,tune = list(
    num.trees = c(1000,1500,2000)
   ,mtry = floor(sqrt(ncol(x.train))*c(1,2))
   )
 ,detailed_names = TRUE ,name_prefix = 'ranger'
)

if(run.below <- FALSE){
glmnet.custom <-  create.Learner('SL.glmnet'
 ,tune = list(
   alpha  = seq(0 ,1 ,length.out=10)  # (0,1)=>(ridge, lasso)
  ,nlambda = seq(0 ,10 ,length.out=10)
   )
 ,detailed_names = TRUE ,name_prefix = 'glmnet'
)}

#ranger.custom <- function(...) SL.ranger(...,num.trees=1000, mtry=5)
#kernelKnn.custom <- function(...) SL.kernelKnn(...,transf_categ_cols=TRUE)

#-----------------------
# SuperLearner Settings
#-----------------------
family   <- 'binomial' #'gaussian'
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
)

#-------------------------------
# Multicore/Parallel Processing
#-------------------------------
if (!require('parallel')) install.packages('parallel'); library(parallel)

cl <- makeCluster(detectCores()-1)

clusterExport(cl, c( listWrappers()

  ,'save.dir'

  ,'SuperLearner','CV.SuperLearner','predict.SuperLearner','cvControl'
  ,'x.train','y.train','x.test','y.test','x.hold','y.hold'
  ,'family','nnls','auc','nnloglik'

  ,'SL.algorithm'
  ,ranger.custom$names,xgboost.custom$names
#  ,glmnet.custom$names

  ))

clusterSetRNGStream(cl, iseed=135)

## Load libraries on workers
clusterEvalQ(cl, {
  library(SuperLearner);library(caret)
  library(ranger);library(xgboost)
  library(glmnet)#;library(randomForest)
#library(kernlab)
#library(arm)
#ibrary(MASS)
#library(klaR)
#ibrary(nnet)
#library(e1071)
})

clusterEvalQ(cl, {

  ensem.nnls <- SuperLearner(Y=y.train ,X=x.train #,verbose=TRUE
    ,family=family,method=nnls,SL.library=SL.algorithm,cvControl=list(V=cvControl)
    );saveRDS(ensem.nnls ,paste0(save.dir,'nnls'))

  ensem.auc <- SuperLearner(Y=y.train ,X=x.train #,verbose=TRUE
    ,family=family,method=auc,SL.library=SL.algorithm,cvControl=list(V=cvControl)
    );saveRDS(ensem.auc ,paste0(save.dir,'auc'))

  ensem.nnloglik <- SuperLearner(Y=y.train ,X=x.train #,verbose=TRUE
    ,family=family,method=nnloglik,SL.library=SL.algorithm,cvControl=list(V=cvControl)
    );saveRDS(ensem.nnloglik ,paste0(save.dir,'nnloglik'))

})

system.time({
  ensem.nnls.cv <- CV.SuperLearner(Y=y.hold ,X=x.hold #,verbose=TRUE
   ,cvControl=list(V=cvControl),innerCvControl=list(list(V=cvControl-1))
   ,family=family ,method=nnls ,SL.library=SL.algorithm ,parallel=cl
  );saveRDS(ensem.nnls.cv ,paste0(save.dir,'nnls.cv'))
})

system.time({
  ensem.auc.cv <- CV.SuperLearner( Y=y.hold ,X=x.hold ,verbose=TRUE
   ,cvControl=list(V=cvControl),innerCvControl=list(list(V=cvControl-1))
   ,family=family ,method=auc ,SL.library=SL.algorithm ,parallel=cl
  );saveRDS(ensem.auc.cv ,paste0(save.dir,'auc.cv'))
})

system.time({
  ensem.nnloglik.cv <- CV.SuperLearner( Y=y.hold ,X=x.hold ,verbose=TRUE
   ,cvControl=list(V=cvControl),innerCvControl=list(list(V=cvControl-1))
   ,family=family ,method=nnloglik ,SL.library=SL.algorithm ,parallel=cl
  );saveRDS(ensem.nnloglik.cv ,paste0(save.dir,'nnloglik.cv'))
})

#stopCluster(cl)

#------------------------------------------
# Read in results form papallel processing
#------------------------------------------
ensem.nnls     <- readRDS(paste0(save.dir,'nnls'));ensem.nnls$times
ensem.auc      <- readRDS(paste0(save.dir,'auc' ));ensem.auc$times
ensem.nnloglik <- readRDS(paste0(save.dir,'nnloglik'));ensem.nnloglik$times

#----------------------
# Risk and Coefficient
#----------------------
compare <- noquote(cbind(
   ensem.nnls$cvRisk,ensem.nnls$coef
  ,ensem.auc$cvRisk,ensem.auc$coef
  ,ensem.nnloglik$cvRisk,ensem.nnloglik$coef
));colnames(compare) <- c(
   'nnls.cvRisk','nnls.coef'
  ,'auc.cvRisk','auc.coef'
  ,'nnloglik.cvRisk','nnloglik.coef'
);write.csv(compare,paste0(save.dir,'compare.csv'))

#compare[,c(1,3,5)];compare[,c(2,4,6)]

#---------------------------------------
# 2D Scatter Plot - Risks & Coefficient
#---------------------------------------
(p2d.risk_coef <- plot_ly( as.data.frame(compare)

  ,x=~ensem.nnls$libraryNames
  ,y=~ensem.nnls$cvRisk, name='risk nnls'
  ,hoverinfo = 'text' ,text = ~paste(
    'function:' ,ensem.nnls$libraryNames
    ,'\nrisk:' ,ensem.nnls$cvRisk)

  ,type='scatter',mode = 'markers'
  ,width=1500 ,height=750 #,margin=5
  ,marker=list( size = 10, opacity = 0.5
    ,line=list( color='black' ,width=1
     #,shape='spline' ,smoothing=1.3
      ))
) %>% add_trace(y = ~ensem.auc$cvRisk, name='risk auc'
                ,hoverinfo = 'text' ,text = ~paste(
                  'function:' ,ensem.auc$libraryNames
                  ,'\nrisk:' ,ensem.auc$cvRisk)
) %>% add_trace(y = ~ensem.nnloglik$cvRisk, name='risk nnloglik'
                ,hoverinfo = 'text' ,text = ~paste(
                  'function:' ,ensem.nnloglik$libraryNames
                  ,'\nrisk:' ,ensem.nnloglik$cvRisk)

) %>% add_trace(y = ~ensem.nnls$coef, name='coef nnls',yaxis='y2'
                ,hoverinfo = 'text' ,text = ~paste(
                  'function:' ,ensem.nnls$libraryNames
                 ,'\ncoefficient:' ,ensem.nnls$coef)
) %>% add_trace(y = ~ensem.auc$coef, name='coef auc',yaxis='y2'
                ,hoverinfo = 'text' ,text = ~paste(
                  'function:' ,ensem.auc$libraryNames
                  ,'\ncoefficient:' ,ensem.auc$coef)
) %>% add_trace(y = ~ensem.nnloglik$coef, name='coef nnloglik',yaxis='y2'
                ,hoverinfo = 'text' ,text = ~paste(
                  'function:' ,ensem.nnloglik$libraryNames
                  ,'\ncoefficient:' ,ensem.nnloglik$coef)

) %>% layout( title=sprintf("Learner/Algorithm Evaluations with Tuning Parameters (SMOTE'd Training Data = %i obs.)", train.obs)
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
));htmlwidgets::saveWidget(
  p2d.risk_coef,paste0(prefix,'.p2d.risk_coef.html')
);file.move(paste0(prefix,'.p2d.risk_coef.html'), output.folder)

#%>% add_lines(y = ~ensem.auc$cvRisk, colors = "black", alpha = 0.2)

#------------
# PREDICTION
#------------
pred.nnls     <- predict.SuperLearner(ensem.nnls     ,x.test ,onlySL=TRUE)
pred.auc      <- predict.SuperLearner(ensem.auc      ,x.test ,onlySL=TRUE)
pred.nnloglik <- predict.SuperLearner(ensem.nnloglik ,x.test ,onlySL=TRUE)

#----------------------------------
# Summary of Predictions by Method
#----------------------------------
prediction.summary <- noquote(cbind(
  summary(pred.nnls$pred)
 ,summary(pred.auc$pred)
 ,summary(pred.nnloglik$pred))
);colnames(prediction.summary) <- c('nnls','auc','nnloglik'
);write.csv(
  prediction.summary,paste0(save.dir,'prediction.summary.csv'))

#pred.method.summary

#----------------------------
# PREDICTION TYPES BY METHOD
#----------------------------
#cat('Classification threshold:', threshold)

pred_type <- function(plist, label=y.test, cutoff=0.5555) {

  ptype <- rep(NA, length(y.test))
  ptype <-
    ifelse(plist >= cutoff & label == 1, "TP",
      ifelse(plist >= cutoff & label == 0, "FP",
        ifelse(plist < cutoff & label == 1, "FN",
          ifelse(plist < cutoff & label == 0, "TN", '??'))))
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
);write.csv(pred.type,paste0(save.dir,'pred.type.csv'))

#prediction.type

#----------------------------------------------------
# 2D scatter Plot of Prediciton by Ensemble Learning
#----------------------------------------------------
if (!require('plotly')) install.packages('plotly');library(plotly)
if (!require('RColorBrewer')) install.packages('RColorBrewer');library(RColorBrewer)

#cat('Classification threshold:', threshold)

(p2d <- plot_ly( as.data.frame(pred.type)
  ,x = ~1:nrow(pred.type)
  ,y = ~pred.type[,'label'] ,name='label'
# OPTIONAL BLEOW -----------------------------
,color = ~pred.type[,'description']
,colors=c('red','blue')
# TOPIONAL ABOVE -----------------------------
  ,hoverinfo = 'text' ,text = ~paste(
      'label:' ,pred.type[,'label']
     ,'\nobservation:' ,pred.type[,'description'])
  ,type='scatter' ,width=1500 ,height=700 #,margin=5
  ,mode = 'markers+lines'
  ,marker = list( size = 10 ,opacity = 0.5
     #,color = pred.type ,colorbar=list(title = "Viridis")
     #,color = colorRampPalette(brewer.pal(12,'Set1'))(2000)
     ,line = list( color = 'black' ,width = 1))
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
  'Predicitons by Ensemble Learning (Threshold = %.2f)', threshold)
  ,xaxis=list(title='observation')
  ,yaxis=list(title='prediction')
  ,plot_bgcolor=backgroundcolor
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
));htmlwidgets::saveWidget(p2d,paste0(prefix,'.p2d.html')
);file.move(paste0(prefix,'.p2d.html'), output.folder)

#----------------------------------------
# 2D SCATTER PLOT SHOWING CLASSIFICAITON
#----------------------------------------
(p2d.class <- plot_ly( as.data.frame(pred.type)
                 ,x = ~1:nrow(pred.type)
                 ,y = ~pred.type[,'label'] #,name='label'

                 ,color = ~pred.type[,'description']
                 ,colors=c('red','blue')

                 ,hoverinfo = 'text' ,text = ~paste(
                   'label:' ,pred.type[,'label']
                   ,'\nobservation:' ,pred.type[,'description'])
                 ,type='scatter' ,width=1500 ,height=700 #,margin=5
                 ,mode = 'markers+lines'
                 ,marker = list( size = 10 ,opacity = 0.5
                   #,color = pred.type ,colorbar=list(title = "Viridis")
                   #,color = colorRampPalette(brewer.pal(12,'Set1'))(2000)
                   ,line = list( color = 'black' ,width = 1))
) %>% add_trace(y = ~pred.type[,'nnls'] ,name='nnls' ,mode = 'markers'
                ,marker = list( size = 10 ,opacity = 0.4
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
                ,marker = list( size = 10 ,opacity = 0.3
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
                ,marker = list( size = 10 ,opacity = 0.2
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
  'Predicitons by Ensemble Learning (Threshold = %.2f)', threshold)
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
,colors=c('red','blue')
,marker = list(size = 10 ,opacity = 0.5
# https://moderndata.plot.ly/create-colorful-graphs-in-r-with-rcolorbrewer-and-plotly/
#,color = colorRampPalette(brewer.pal(12,'Set3'))(floor(train.obs/12))
#,colorscale = c(brewer.pal(11,'RdBu')[1],brewer.pal(11,'RdBu')[11]),showscale = TRUE
,line = list( color = 'black' ,width = 0.5))
) %>% add_markers(
) %>% layout( title=sprintf(
  'Predictions by Ensemble Learning (Threshold = %.2f)', threshold)
  ,scene = list(
    xaxis = list(title='auc',showbackground=FALSE
                 ,backgroundcolor=backgroundcolor ,gridcolor=gridcolor
                 ,zerolinecolor='rgb(0,0,0)')
   ,yaxis = list(title='nnls',showbackground=FALSE
                 ,backgroundcolor=backgroundcolor,gridcolor=gridcolor
                 ,zerolinecolor='rgb(0,0,0)')
   ,zaxis = list(title='nnloglik',showbackground=FALSE
                 ,backgroundcolor=backgroundcolor,gridcolor=gridcolor
                 ,zerolinecolor='rgb(0,0,0)')
   ,camera = list(
         up=list(x=0, y=0, z=1)
        ,center=list(x=0, y=0, z=0)
        ,eye=list(x=1.75, y=-1.25, z=0.5)  )
))
);htmlwidgets::saveWidget(p3d,paste0(prefix,'.p3d.html')
);file.move(paste0(prefix,'.p3d.html'), output.folder)

#------------------
# CONSUSION MATRIX
#------------------
# Converting probabilities into classification
pred.type$nnls.converted     <- ifelse(pred.nnls$pred     >= threshold,1,0)
pred.type$auc.converted      <- ifelse(pred.auc$pred      >= threshold,1,0)
pred.type$nnloglik.converted <- ifelse(pred.nnloglik$pred >= threshold,1,0)

# Confusion Matrix
cm.nnls <- confusionMatrix(factor(pred.type$nnls.converted ), factor(y.test)
);saveRDS(cm.nnls ,paste0(save.dir,'.cm.nnls')
);cat('Mean Square Error (NNLS) = ',mse.nnls <- mean((y.test-pred.nnls$pred)^2)
);cm.nnls

cm.auc <- confusionMatrix(factor(pred.type$auc.converted), factor(y.test)
);saveRDS(cm.auc ,paste0(save.dir,'.cm.auc')
);cat('Mean Square Error (AUC) = ',mse.auc <- mean((y.test-pred.auc$pred)^2)
);cm.auc

cm.nnloglik <- confusionMatrix(factor(pred.type$nnloglik.converted ), factor(y.test)
);saveRDS(cm.nnloglik ,paste0(save.dir,'.cm.nnloglik')
);cat('Mean Square Error (NNloglik) = ',mse.nnloglik <- mean((y.test-pred.nnloglik$pred)^2)
);cm.nnloglik

pred.accuracy <- noquote(cbind(
  MSE=c(mse.nnls,mse.auc,mse.nnloglik)
 ,Accuracy=c(
    cm.nnls$overall['Accuracy']
   ,cm.auc$overall['Accuracy']
   ,cm.nnloglik$overall['Accuracy']
  )
)
);rownames(pred.accuracy) <- c('nnls','auc','nnloglik'
);saveRDS(pred.accuracy ,paste0(save.dir,'.pred.accuracy')
);pred.accuracy

#-------------------------
# CROSS VALIDATON OBJECTS
#-------------------------
ensem.nnls.cv     <- readRDS(paste0(save.dir,'nnls.cv'));ensem.nnls.cv$times
ensem.auc.cv      <- readRDS(paste0(save.dir,'auc.cv' ));ensem.auc.cv$times
ensem.nnloglik.cv <- readRDS(paste0(save.dir,'nnloglik.cv'));ensem.nnloglik.cv$times

ensem.nnls.cv$AllSL$'2'$coef
ensem.nnls.cv$AllSL$'2'$cvRisk
plot(x=colnames(ensem.nnls.cv$coef), y=ensem.nnls.cv$coef)


summary(ensem.nnls.cv)
table(simplify2array(ensem.nnls.cv$whichDiscreteSL))

summary(ensem.auc.cv)
table(simplify2array(ensem.auc.cv$whichDiscreteSL))

summary(ensem.nnloglik.cv)
table(simplify2array(ensem.nnloglik.cv$whichDiscreteSL))

#----------
# Stacking
#----------
ensem.nnls.cv.stacking <- plot(ensem.nnls.cv)+theme_bw();ensem.nnls.cv.stacking
ensem.auc.cv.stacking <- plot(ensem.auc.cv)+theme_bw();ensem.auc.cv.stacking
ensem.nnloglik.cv.stacking <- plot(ensem.nnloglik.cv)+theme_bw();ensem.nnloglik.cv.stacking

#----------------------------------------------------------
# CROSS VALIDATION - ROC CURVE
# (receiver operating characteristic curve)
#
# It plots TPR vs. FPR at different classification thresholds.
# https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc
#----------------------------------------------------------
if (!require('ck37r')) install.packages('ck37r') ;library(ck37r)

(ensem.nnls.cv.roc <- cvsl_plot_roc(ensem.nnls.cv)
  );htmlwidgets::saveWidget(ensem.nnls.cv.roc
    ,paste0(save.dir,'nnls.cv.roc'))

(ensem.auc.cv.roc <- cvsl_plot_roc(ensem.auc.cv)
);htmlwidgets::saveWidget(ensem.auc.cv.roc
    ,paste0(save.dir,'auc.cv.roc'))

(ensem.nnloglik.cv.roc <- cvsl_plot_roc(ensem.nnloglik.cv)
  );htmlwidgets::saveWidget(ensem.nnloglik.cv.roc
    ,paste0(save.dir,'.nnloglik.cv.roc'))

par(mfrow=c(1,3),mar=c(1,1,1,1))
cvsl_plot_roc(ensem.nnls.cv)
cvsl_plot_roc(ensem.auc.cv)
cvsl_plot_roc(ensem.nnloglik.cv)
par(mfrow=c(1,1),mar=c(1,1,1,1))

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

cvsl_weights(ensem.nnls.cv)
cvsl_weights(ensem.auc.cv)
cvsl_weights(ensem.nnloglik.cv)

cvsl_auc(ensem.nnls.cv)
cvsl_auc(ensem.auc.cv)
cvsl_auc(ensem.nnloglik.cv)

stopCluster(cl)

