
#import pandas as pd
import numpy as np
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


import pandas as pd


# Reading data from csv file
data = pd.read_csv("URL.csv",encoding='latin-1')
data.head()

# Labels
y = data["label"]

# Features
url_list = data["url"]

# Using Tokenizer
vectorizer = TfidfVectorizer()

# Store vectors into X variable as Our XFeatures
X = vectorizer.fit_transform(url_list)


# Split into training and testing dataset 80:20 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Building using logistic regression
logit = LogisticRegression()
logit.fit(X_train, y_train)


# Accuracy of Our Model
print("Accuracy of our model is: ",logit.score(X_test, y_test))


X_predict = vectorizer.transform(["www.buyfakebillsonlinee.com"])


logit.predict(X_predict)[0]


from flask import Flask, request , jsonify
from flask_restful import Resource, Api
#from flask import Blueprint
from json import dumps
#api_bp = Blueprint('api', __name__)
#from flask.ext.jsonpify import jsonify
from flask_cors import CORS
import traceback

app = Flask(__name__)
CORS(app)
api = Api(app)


class urlDetection(Resource):
    def post(self):		
        try:
            test_url = request.get_json()
            X_predict = vectorizer.transform([test_url['url']])
            result_predict = logit.predict(X_predict)[0]

        #if not json_data:
        #       return {'message': 'No input data provided'}, 400
            return {'status': 'success', 'response': str(result_predict)}, 200
        except:
            return jsonify({'trace': traceback.format_exc()})
api.add_resource(urlDetection, '/api/url-detection')
#api.add_resource(saveTrainData, '/api/retrainmodel')  

if __name__ == '__main__':
     app.run(host='0.0.0.0',port='5000')

