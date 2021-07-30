# Kevin Blum
# CS 643
# Load ML Model And Predict Labels From Given Input

import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.classification import LogisticRegression, OneVsRest, OneVsRestModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import FeatureHasher
from pyspark.mllib.evaluation import MulticlassMetrics

# Command Line arguments should be in order of data file you wanted to be predicted and then the s3 bucket you are loading the ML model from. In this case I used "s3://kevs-643-bucket/oneVsModel"

spark = SparkSession \
    .builder \
    .appName("MLProject") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

# Take file as input
data = sys.argv[1]

# Read input file
dataFrame = spark.read.option("header" , "True").option("inferSchema","True").option("sep",";").csv(data)

# Featurize Data so it is suitable for model
fHash = FeatureHasher(inputCols=['"""""fixed acidity""""','""""volatile acidity""""','""""citric acid""""','""""residual sugar""""',\
                                  '""""chlorides""""','""""free sulfur dioxide""""','""""total sulfur dioxide""""','""""density""""'\
                                  ,'""""pH""""','""""sulphates""""','""""alcohol""""'],outputCol="features")
                            
                            

# Transform dataframe
featDF = fHash.transform(dataFrame)
featDF = featDF.withColumnRenamed('""""quality"""""','label')

# load model
# "s3://kevs-643-bucket/oneVsModel"

bucket = sys.argv[2]
ovr_model = OneVsRestModel.load(bucket)

# Make Predictions
labelPredictions = ovr_model.transform(featDF)

# Put labelPredictions in correct format to be used by MulticlassMetrics and Fmeasure
labelPredictions = labelPredictions['label','prediction']
labelPredictions.show(5)
labelPredictions = labelPredictions.withColumn('label',col('label').cast('Float'))
labelPredictions = labelPredictions.withColumn('prediction',col('prediction').cast('Float'))
labelPredictions = labelPredictions.rdd



# Get Metrics
measures = MulticlassMetrics(labelPredictions)

# F1 Score
f1 = measures.fMeasure()
print("F1 Score = %s" % f1)
