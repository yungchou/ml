if (!require('h2o')) install.packages('h2o'); library(h2o)

h2o.init(nthreads=-1)
# H2O Demo
#demo(h2o.gbm)
#demo(h2o.gbm)

#==============
# Demo h2o.glm
#==============
prostate.hex = h2o.uploadFile(
	path = system.file("extdata", "prostate.csv", package="h2o"), 
	destination_frame = "prostate.hex")

summary(prostate.hex)

prostate.km = h2o.kmeans(prostate.hex, k = 10, x = c("AGE","RACE","GLEASON","CAPSULE","DCAPS"))
print(prostate.km)

prostate.data = as.data.frame(prostate.hex)

# prostate.clus = as.data.frame(prostate.km@model$cluster)
 
# Plot categorized data
# if(!"fpc" %in% rownames(installed.packages())) install.packages("fpc")
# if("fpc" %in% rownames(installed.packages())) {
#  library(fpc)
 
#  par(mfrow=c(1,1))
#  plotcluster(prostate.data, prostate.clus[,1])
#  title("K-Means Classification for k = 10")
# }
 
# if(!"cluster" %in% rownames(installed.packages())) install.packages("cluster")
# if("cluster" %in% rownames(installed.packages())) {
#  library(cluster)
#  clusplot(prostate.data, prostate.clus[,1], color = TRUE, shade = TRUE)
# }
# pairs(prostate.data[,c(2,3,7,8)], col=prostate.clus[,1])
 
# Plot k-means centers
par(mfrow = c(1,2))

prostate.ctrs = as.data.frame(prostate.km@model$centers)

plot(prostate.ctrs[,1:2])
plot(prostate.ctrs[,3:4])
title("K-Means Centers for k = 10", outer = TRUE, line = -2.0)

#==============
# Demo h2o.gbm
#==============
# This is a demo of H2O's GBM function
# It imports a data set, parses it, and prints a summary
# Then, it runs GBM on a subset of the dataset
# Note: This demo runs H2O on localhost:54321

if (!require('h2o')) install.packages('h2o'); library(h2o)
h2o.init()

prostate.hex = h2o.uploadFile(
	path = system.file("extdata", "prostate.csv", package="h2o"), 
	destination_frame = "prostate.hex")

summary(prostate.hex)

prostate.gbm = h2o.gbm(
	x = setdiff(colnames(prostate.hex), "CAPSULE"), 
	y = "CAPSULE", 
	training_frame = prostate.hex, ntrees = 10, max_depth = 5, learn_rate = 0.1)

print(prostate.gbm)

prostate.gbm2 = h2o.gbm(
	x = c("AGE", "RACE", "PSA", "VOL", "GLEASON"), 
	y = "CAPSULE", 
	training_frame = prostate.hex, ntrees = 10, max_depth = 8, min_rows = 10, learn_rate = 0.2)

print(prostate.gbm2)

#-------------------------------------------------------
# This is a demo of H2O's GBM use of default parameters 
# on iris dataset (three classes)
#-------------------------------------------------------

iris.hex = h2o.uploadFile(
	path = system.file("extdata", "iris.csv", package="h2o"), 
	destination_frame = "iris.hex")

# Model
iris.gbm = h2o.gbm(x = 1:4, y = 5, training_frame = iris.hex)

print(iris.gbm)

