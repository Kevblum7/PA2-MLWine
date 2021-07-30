# Kevin Blum
# CS 643
# Train ML Model
# Logistic Regression One Vs Rest

# command line arguments should be taken in as Training Data, Testing Data, and name of s3 bucket to save your model

import sys
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, OneVsRest
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import FeatureHasher

spark = SparkSession \
    .builder \
    .appName("MLProject") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()


# Read CSV file

# "s3://kevs-643-bucket/TrainingDataset.csv" is what I used as my input for training data
trainData = sys.argv[1]
dataFrame=spark.read.option("header" , "True").option("inferSchema","True").option("sep",";").csv(trainData)


# Feature Hasher adds vectorized features to be used in model
fHash = FeatureHasher(inputCols=['"""""fixed acidity""""','""""volatile acidity""""','""""citric acid""""','""""residual sugar""""',\
                                  '""""chlorides""""','""""free sulfur dioxide""""','""""total sulfur dioxide""""','""""density""""'\
                                  ,'""""pH""""','""""sulphates""""','""""alcohol""""'],outputCol="features")

# featurized data with features column 
featDF = fHash.transform(dataFrame)

# ML model needs label column. Switch quality to 'label'
featDF =  featDF.withColumnRenamed('""""quality"""""','label')

# train base logistic regression classifier
lrClass = LogisticRegression(maxIter=10, tol=1E-6, fitIntercept=True)

oneVs = OneVsRest(classifier=lrClass)

# train multi-class model
oneVsModel = oneVs.fit(featDF)

# gather test data or 'Validation' dataset
# "s3://kevs-643-bucket/ValidationDataset.csv" is what I used for my input of validation data
testData = sys.argv[2]
dataFrameV=spark.read.option("header" , "True").option("inferSchema","True").option("sep",";").csv(testData)

# do same data manipulations for model
fHashValid = FeatureHasher(inputCols=['"""""fixed acidity""""','""""volatile acidity""""','""""citric acid""""','""""residual sugar""""',\
                                  '""""chlorides""""','""""free sulfur dioxide""""','""""total sulfur dioxide""""','""""density""""'\
                                  ,'""""pH""""','""""sulphates""""','""""alcohol""""'],outputCol="features")

featDFV = fHashValid.transform(dataFrameV)
featDFV = featDFV.withColumnRenamed('""""quality"""""','label')

# score with test data
labelPredictions = oneVsModel.transform(featDFV)

evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
accuracy = evaluator.evaluate(labelPredictions)
print("Accuracy = %g " % (accuracy))

# Save model to my s3 bucket to be later downloaded
# "s3://kevs-643-bucket/oneVsModel" s3 bucket I saved my model

bucketModel = sys.argv[3]
oneVsModel.save(bucketModel)
