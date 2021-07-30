CS 643 -Programing Assignment 2 

ML Wine Quality Prediction 

Using Amazon EMR to launch Spark Applications using MLlib to train and fit machine learning model for wine quality prediction.

“TrainML_OVR.py “is code to train model and save it

“PredictorApp.py” loads model and uses it for data to make quality predictions. Outputs F1 score


Training the model

- Launch Amazon EMR cluster with 4 nodes and spark applications. Create S3 bucket to store files to be be downloaded for training and predicting applications. 
Launch master ec2 instance by connecting with ssh and launch training model: 

spark-submit \ 
—class sys \ 
<path to TrainML_OVR.py> \
<path to training data file>  <path to testing data file> < path to store ml model>

- All of the paths used were from S3 buckets

Launching Prediction Application 

- launch application using your test data for command line argument and path to saved model. 

spark-submit \
--class sys \
<Path to PredictorApp.py> \
<Path to test data> <Path to saved model>




