#-----------------------------------------------------
# TRAINING AND VALIDATING MODEL WITH A CARTESIAN GRID
#-----------------------------------------------------
# (1) GBM hyperparamters
gbm_params <- list(
   ntrees          = 100
  ,learn_rate      = c(0.01, 0.1)
  ,max_depth       = c(3, 5, 9)
  ,sample_rate     = c(0.8, 1.0)
  ,col_sample_rate = c(0.2, 0.5, 1.0)
  )

# Train and validate a cartesian grid of GBMs
gbm_grid <- h2o.grid('gbm', x = x, y = y, grid_id = 'gbm.grid'
  ,seed = seed, hyper_params = gbm_params ,nfolds=nfolds
  ,training_frame = training_frame, validation_frame = valid_frame
  )

# Get the grid results, sorted by validation AUC
gbm_grid_perf <- h2o.getGrid(grid_id='gbm.grid',sort_by='auc',decreasing=TRUE)
print(gbm_grid_perf)

# Grab the top GBM model, chosen by validation AUC
best_gbm <- h2o.getModel(gbm_grid_perf@model_ids[[1]])

# Now let's evaluate the model performance on a test set
best_gbm_perf <- h2o.performance(model = best_gbm,newdata = testing_frame)
h2o.auc(best_gbm_perf)

# Look at the hyperparamters for the best model
print(best_gbm@model[['model_summary']])

# (2) Random Forest Hyperparameters
rf_params <- list(
   ntrees    = c(50, 100, 120)
  ,max_depth = c(40, 60)
  ,min_rows  = c(1, 2)
  )

rf_grid <- h2o.grid('randomForest', grid_id = 'rf.grid'
      ,seed = seed, hyper_params = rf_params ,nfolds=nfolds
      ,training_frame = training_frame, validation_frame = valid_frame
      )

rf_r2 <- h2o.getGrid(rf_grid@grid_id, sort_by = 'r2', decreasing = TRUE)



# (3) GLM Hyperparameters
solvers <- c("IRLSM", "L_BFGS", "COORDINATE_DESCENT_NAIVE", "COORDINATE_DESCENT")
families <- c("gaussian", "poisson", "gamma")
gaussianLinks <- c("identity", "log", "inverse")
poissonLinks <- c("log")
gammaLinks <- c("identity", "log", "inverse")
gammaLinks_CD <- c("identity", "log")
allGrids <- lapply(solvers, function(solver){
  lapply(families, function(family){
    if(family == "gaussian")theLinks <- gaussianLinks
    else if(family == "poisson")theLinks <- poissonLinks
    else{
      if(solver == "COORDINATE_DESCENT")theLinks <- gammaLinks_CD
      else theLinks = gammaLinks
    }
    lapply(theLinks, function(link){
      grid_id = paste("GLM", solver, family, link, sep="_")
      h2o.grid("glm", grid_id = grid_id,
               hyper_params = list(
                 alpha = c(0, 0.1, 0.5, 0.99)
               ),
               x = x, y = y, training_frame = train,
               nfolds = 10,
               lambda_search = TRUE,
               solver = solver,
               family = family,
               link = link,
               max_iterations = 100
      )
    })
  })
})

# (4) Deep Learning Hyperparameters
dl_params <- list(
  epochs=1000,
  hidden=list(c(32,32,32),c(64,64)),
  input_dropout_ratio=c(0,0.05),
  rate=c(0.01,0.02),
  rate_annealing=c(1e-8,1e-7,1e-6)
)

dl_grid <- h2o.grid('deeplearning', x=x,y=y, grid_id='dl.grid', nfolds=nfolds,
  training_frame=training_frame,  validation_frame=validation_frame,
  hyper_params=dl_params,
  stopping_metric="misclassification", #'MSE'
  stopping_tolerance=1e-2, ## stop when misclassification does not improve by >=1% for 2 scoring events
  stopping_rounds=2,
  score_validation_samples=10000, ## downsample validation set for faster scoring
  score_duty_cycle=0.025, ## don't score more than 2.5% of the wall time
  adaptive_rate=F, ## manually tuned learning rate
  momentum_start=0.5, ## manually tuned momentum
  momentum_stable=0.9,
  momentum_ramp=1e7,
  l1=1e-5,
  l2=1e-5,
  activation=c("Rectifier"), #'Tanh'
  max_w2=10 ## can help improve stability for Rectifier
)

dl.grid <- h2o.getGrid('dl.grid',sort_by="err",decreasing=FALSE)

m <- h2o.deeplearning(x=x, y=y, training_frame, nfolds = nfolds, seed=seed,
                      model_id = 'dl',
                      activation = 'Tanh',
                      l2 = 0.00001, #1e-05
                      hidden = c(162,162),
                      stopping_metric = 'MSE',
                      stopping_tolerance = 0.0005,
                      stopping_rounds = 5,
                      epochs = 2000,
                      train_samples_per_iteration = 0,
                      score_interval = 3
)