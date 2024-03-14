import json
from collections import defaultdict
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
import numpy
import random
import gzip
import dateutil.parser
import math

answers = {}


### QUESTION 1
f = gzip.open("fantasy_10000.json.gz")
dataset = []
for l in f:
    dataset.append(json.loads(l))

max_len = 0
for datum in dataset:
    max_len = max(max_len, len(datum['review_text']))

def feature(datum):
    return len(datum['review_text']) / max_len

X = numpy.array([feature(x) for x in dataset]).reshape(-1, 1)
Y = [dataset[x]['rating'] for x in range(len(dataset))]

model = linear_model.LinearRegression()
model.fit(X, Y)

theta = [model.intercept_]
theta += [model.coef_[0]]
y_pred = model.predict(X)
MSE = mean_squared_error(Y, y_pred)
answers['Q1'] = [theta[0], theta[1], MSE]


### QUESTION 2
for d in dataset:
    t = dateutil.parser.parse(d['date_added'])
    d['parsed_date'] = t

def feature(datum):
    feature_vec = [0] * 19
    feature_vec[0] = 1
    feature_vec[1] = len(datum['review_text']) / max_len
    weekday = datum['parsed_date'].weekday() + 1
    if weekday != 1:
        feature_vec[weekday] = 1
    month = datum['parsed_date'].month + 6
    if month != 7:
        feature_vec[month] = 1
    return feature_vec

X = [feature(x) for x in dataset]
Y = [dataset[x]['rating'] for x in range(len(dataset))]

model = linear_model.LinearRegression(fit_intercept=False)
model.fit(X, Y)

answers['Q2'] = [X[0], X[1]]


### QUESTION 3
def feature3(datum):
    return [1, 
            len(datum['review_text']) / max_len, 
            datum['parsed_date'].weekday(), 
            datum['parsed_date'].month]

X3 = [feature3(x) for x in dataset]
Y3 = [dataset[x]['rating'] for x in range(len(dataset))]

model3 = linear_model.LinearRegression(fit_intercept=False)
model3.fit(X3, Y3)

y_pred2 = model.predict(X)
mse2 = mean_squared_error(Y, y_pred2)
y_pred3 = model3.predict(X3)
mse3 = mean_squared_error(Y3, y_pred3)
answers['Q3'] = [mse2, mse3]


### QUESTION 4
random.seed(0)
random.shuffle(dataset)

X2 = [feature(d) for d in dataset]
X3 = [feature3(d) for d in dataset]
Y = [d['rating'] for d in dataset]

train2, test2 = X2[:len(X2)//2], X2[len(X2)//2:]
train3, test3 = X3[:len(X3)//2], X3[len(X3)//2:]
trainY, testY = Y[:len(Y)//2], Y[len(Y)//2:]

model2_1 = linear_model.LinearRegression(fit_intercept=False)
model2_1.fit(train2, trainY)
model3_1 = linear_model.LinearRegression(fit_intercept=False)
model3_1.fit(train3, trainY)

y_pred2_1 = model2_1.predict(test2)
test_mse2 = mean_squared_error(testY, y_pred2_1)
y_pred3_1 = model3_1.predict(test3)
test_mse3 = mean_squared_error(testY, y_pred3_1)
answers['Q4'] = [test_mse2, test_mse3]


### QUESTION 5
f = open("beer_50000.json")
dataset = []
for l in f:
    dataset.append(eval(l))

max_len = 0
for datum in dataset:
    max_len = max(max_len, len(datum['review/text']))

def feature(datum):
    return [1, len(datum['review/text']) / max_len]

X = [feature(x) for x in dataset]
y = [1 if x['review/overall'] >= 4 else 0 for x in dataset]

model = linear_model.LogisticRegression(fit_intercept=False, class_weight='balanced', C=1.0)
model.fit(X, y)

y_pred = model.predict(X)
y_pred = y_pred.tolist()
TP = sum([(a and b) for (a, b) in zip(y_pred, y)])
TN = sum([(not a and not b) for (a, b) in zip(y_pred, y)])
FP = sum([(a and not b) for (a, b) in zip(y_pred, y)])
FN = sum([(not a and b) for (a, b) in zip(y_pred, y)])

BER = 1 - 0.5 * (TP / (TP + FN) + TN / (TN + FP))

answers['Q5'] = [TP, TN, FP, FN, BER]


### QUESTION 6
scores = model.decision_function(X)
score_labels = list(zip(scores, y))
score_labels.sort(reverse=True)
sorted_labels = [x[1] for x in score_labels]

precs = []

for k in [1,100,1000,10000]:
    precision_k = sum(sorted_labels[:k]) / k
    precs.append(precision_k)

answers['Q6'] = precs


### QUESTION 7
def feature2(datum):
    return [1, 
            len(datum['review/text']) / max_len,
            datum['beer/ABV'],
            datum['review/aroma'],
            datum['review/palate'],
            datum['review/taste'],
            1 if 'stout' in datum['beer/style'].lower() else 0,
            1 if 'PA' in datum['beer/style'] else 0,
            1 if 'ale' in datum['beer/style'].lower() else 0,
            1 if 'porter' in datum['beer/style'].lower() else 0]

X2 = [feature2(x) for x in dataset]
y2 = [1 if x['review/overall'] >= 4 else 0 for x in dataset]

model2 = linear_model.LogisticRegression(fit_intercept=False, class_weight='balanced', C=1.0, max_iter=500)
model2.fit(X2, y2)

y_pred2 = model2.predict(X2)
y_pred2 = y_pred2.tolist()
TP2 = sum([(a and b) for (a, b) in zip(y_pred2, y2)])
TN2 = sum([(not a and not b) for (a, b) in zip(y_pred2, y2)])
FP2 = sum([(a and not b) for (a, b) in zip(y_pred2, y2)])
FN2 = sum([(not a and b) for (a, b) in zip(y_pred2, y2)])

its_test_BER = 1 - 0.5 * (TP2 / (TP2 + FN2) + TN2 / (TN2 + FP2))

answers['Q7'] = ["I decided to add the features that I thought would intuitively be some indicators to help decide if a beer is good or bad. One feature that can tell about how a beer will taste is the ABV and the style. For the different styles\
                  of beer, I listed them by the proportion of the style that had review/overall >= 4. I noticed that stouts, IPAs, Ales, and Porters were typically well reviewed and popular styles in general. Thus, I did a one-hot encoding for these\
                  styles; if a beer wasn't any of these styles, then the one-hot encoding would be all zeros. I thought that the review rating subcategories could help predict the overall score. Thus, I added the features for aroma, palate, and \
                  taste ratings which I believed to be most indicative of the overall score. With these features, I achieved a BER of 0.1760 vs the previous BER of 0.4683", its_test_BER]
