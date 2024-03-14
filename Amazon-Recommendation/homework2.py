import random
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from collections import defaultdict
import gzip
import sys
from math import e
from datetime import datetime

answers = {}

def parseData(fname):
    for l in open(fname):
        yield eval(l)

data = list(parseData("beer_50000.json"))

random.seed(0)
random.shuffle(data)

dataTrain = data[:25000]
dataValid = data[25000:37500]
dataTest = data[37500:]

yTrain = [d['beer/ABV'] > 7 for d in dataTrain]
yValid = [d['beer/ABV'] > 7 for d in dataValid]
yTest = [d['beer/ABV'] > 7 for d in dataTest]

max_review_len = 0
for d in dataTrain:
    max_review_len = max(max_review_len, len(d['review/text']))

categoryCounts = defaultdict(int)
for d in data:
    categoryCounts[d['beer/style']] += 1
categories = [c for c in categoryCounts if categoryCounts[c] > 1000]
catID = dict(zip(list(categories),range(len(categories))))

def feat(d, includeCat = True, includeReview = True, includeLength = True):
    feats = []
    if includeCat:
        feats += [0] * len(catID)
        if d['beer/style'] in catID: feats[catID[d['beer/style']]] = 1
    
    if includeReview:
        feats += [d['review/appearance'], d['review/aroma'], d['review/overall'], d['review/palate'], d['review/taste']]

    if includeLength:
        feats += [len(d['review/text']) / max_review_len]
        
    return feats

def pipeline(reg, includeCat = True, includeReview = True, includeLength = True, max_iter = 100):
    xTrain = [feat(d, includeCat=includeCat, includeReview=includeReview, includeLength=includeLength) for d in dataTrain]
    xValid = [feat(d, includeCat=includeCat, includeReview=includeReview, includeLength=includeLength) for d in dataValid]
    xTest = [feat(d, includeCat=includeCat, includeReview=includeReview, includeLength=includeLength) for d in dataTest]
    model = linear_model.LogisticRegression(C=reg, class_weight='balanced', max_iter=max_iter)
    model.fit(xTrain, yTrain)

    y_pred_valid = model.predict(xValid)
    y_pred_test = model.predict(xTest)

    ber_valid = sum(yValid != y_pred_valid) / len(yValid)
    ber_test = sum(yTest != y_pred_test) / len(yTest)

    return model, ber_valid, ber_test
    

### Question 1
mod, validBER, testBER = pipeline(10, True, False, False)
answers['Q1'] = [validBER, testBER]


### Question 2
mod, validBER, testBER = pipeline(10, True, True, True, 500)
answers['Q2'] = [validBER, testBER]


### Question 3
lowest_ber = sys.maxsize
bestC = -1
for c in [0.001, 0.01, 0.1, 1, 10]:
    mod, validBER, testBER = pipeline(c, True, True, True, 500)
    if validBER < lowest_ber: bestC = c

mod, validBER, testBER = pipeline(bestC, True, True, True, 500)
answers['Q3'] = [bestC, validBER, testBER]


### Question 4
mod, validBER, testBER_noCat = pipeline(1, False, True, True, 500)
mod, validBER, testBER_noReview = pipeline(1, True, False, True, 500)
mod, validBER, testBER_noLength = pipeline(1, True, True, False, 500)
answers['Q4'] = [testBER_noCat, testBER_noReview, testBER_noLength]


### Question 5
path = "amazon_reviews_us_Musical_Instruments_v1_00.tsv"
f = open(path, 'rt', encoding="utf8")

header = f.readline()
header = header.strip().split('\t')

dataset = []

pairsSeen = set()

for line in f:
    fields = line.strip().split('\t')
    d = dict(zip(header, fields))
    ui = (d['customer_id'], d['product_id'])
    if ui in pairsSeen:
        print("Skipping duplicate user/item:", ui)
        continue
    pairsSeen.add(ui)
    d['star_rating'] = int(d['star_rating'])
    d['helpful_votes'] = int(d['helpful_votes'])
    d['total_votes'] = int(d['total_votes'])
    dataset.append(d)

dataTrain = dataset[:int(len(dataset)*0.9)]
dataTest = dataset[int(len(dataset)*0.9):]

# Feel free to keep or discard

usersPerItem = defaultdict(set) # Maps an item to the users who rated it
itemsPerUser = defaultdict(set) # Maps a user to the items that they rated
itemNames = {}
ratingDict = {} # To retrieve a rating for a specific user/item pair
reviewsPerUser = defaultdict(list)

for d in dataTrain:
    user,item = d['customer_id'], d['product_id']
    usersPerItem[item].add(user)
    itemsPerUser[user].add(item)
    reviewsPerUser[user].append(d)

for d in dataset:
    user,item = d['customer_id'], d['product_id']
    ratingDict[(user,item)] = d['star_rating']
    itemNames[item] = d['product_title']

userAverages = {u: sum([ratingDict[(u,i)] for i in itemsPerUser[u]]) / len(itemsPerUser[u]) for u in itemsPerUser}
itemAverages = {i: sum([ratingDict[(u,i)] for u in usersPerItem[i]]) / len(usersPerItem[i]) for i in usersPerItem}

ratingMean = sum([d['star_rating'] for d in dataTrain]) / len(dataTrain)

def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    if denom == 0: return 0
    return numer / denom

def mostSimilar(i, N):
    similarities = []
    users = usersPerItem[i]
    for j in usersPerItem:
        if i == j: continue
        sim = Jaccard(users, usersPerItem[j])
        similarities.append((sim, j))
    similarities.sort(reverse=True)
    return similarities[:N]

query = 'B00KCHRKD6'
ms = mostSimilar(query, 10)
answers['Q5'] = ms


### Question 6
def MSE(y, ypred):
    return sum([(y[i] - ypred[i])**2 for i in range(len(y))]) / len(y)

def predictRating(user,item):
    ratings = []
    similarities = []
    for d in reviewsPerUser[user]:
        i2 = d['product_id']
        if i2 == item: continue
        ratings.append(d['star_rating'] - itemAverages[i2])
        similarities.append(Jaccard(usersPerItem[item], usersPerItem[i2]))
    if sum(similarities) > 0:
        weightedRatings = [(x*y) for x, y in zip(ratings, similarities)]
        return itemAverages[item] + sum(weightedRatings) / sum(similarities)
    else:
        if item in itemAverages: 
            return itemAverages[item]
        else: 
            return ratingMean

alwaysPredictMean = [ratingMean] * len(dataTest)
simPredictions = [predictRating(d['customer_id'], d['product_id']) for d in dataTest]
labels = [d['star_rating'] for d in dataTest]
answers['Q6'] = MSE(simPredictions, labels)


### Question 7
def decay(l, t1, t2):
    t1_unix = datetime.strptime(t1, '%Y-%m-%d').timestamp()
    t2_unix = datetime.strptime(t2, '%Y-%m-%d').timestamp()
    t_diff = abs(t2_unix - t1_unix)
    return e ** (-l * t_diff)

def predictRating(user, item, time):
    ratings = []
    similarities = []
    decays = []
    for d in reviewsPerUser[user]:
        i2 = d['product_id']
        if i2 == item: continue
        ratings.append(d['star_rating'] - itemAverages[i2])
        similarities.append(Jaccard(usersPerItem[item], usersPerItem[i2]))
        decays.append(decay(1, d['review_date'], time))
    if sum(similarities) > 0:
        weightedRatings = [(x*y*z) for x, y, z in zip(ratings, similarities, decays)]
        return itemAverages[item] + sum(weightedRatings) / sum(similarities)
    else:
        if item in itemAverages: 
            return itemAverages[item]
        else: 
            return ratingMean

decayPredictions = [predictRating(d['customer_id'], d['product_id'], d['review_date']) for d in dataTest]
itsMSE = MSE(decayPredictions, labels)

answers['Q7'] = ["The heuristic behind the decay function is that reviews made closer in time to that of the target item (argument passed into predictRating) will be \
                  more indicitive of the correct rating. Some factors that could lead to this are consumer preferences changing over time, products iterating and \
                  improving over time, or to simply filter out older reviews that could introduce irrelevancies/noise in our prediction. I decided to go with the \
                  f(abs(t_u,i - t_u,j)) decay function. Applying this function as shown in the equation on the homework doc, I was able to slightly reduce the MSE \
                  over the trivial decay function (Question 6 MSE)", itsMSE]

f = open("answers_hw2.txt", 'w')
f.write(str(answers) + '\n')
f.close()

