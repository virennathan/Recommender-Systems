# %%
import gzip
from collections import defaultdict
import math
import scipy.optimize
from sklearn import svm
import numpy
import string
import random
import string
from sklearn import linear_model

# %%
def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N

# %%
def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)

# %%
def readJSON(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        d = eval(l)
        u = d['userID']
        g = d['gameID']
        yield u,g,d

# %%
answers = {}

# %%
# Some data structures that will be useful

# %%
allHours = []
for l in readJSON("train.json.gz"):
    allHours.append(l)

# %%
dataTrain = [(u, g) for u, g, _ in allHours[:165000]]
dataValid = [(u, g, 1) for u, g, _ in allHours[165000:]]

# %%
##################################################
# Play prediction                                #
##################################################

# %%
# Any other preprocessing...
playedByUser = defaultdict(set)
allGames = set()

for u, g, _ in allHours:
    playedByUser[u].add(g)
    allGames.add(g)

negDataValid = [None] * len(dataValid)
for i in range(len(dataValid)):
    u = dataValid[i][0]
    g = random.sample(allGames - playedByUser[u], 1)[0]
    playedByUser[u].add(g)
    negDataValid[i] = (u, g, 0)

dataValid += negDataValid

# %%
gameCount = defaultdict(int)
totalPlayed = 0

for _, game in dataTrain:
  gameCount[game] += 1
  totalPlayed += 1

mostPopular = [(gameCount[x], x) for x in gameCount]
mostPopular.sort()
mostPopular.reverse()

# %%
gamesPerUser = defaultdict(set)
usersPerGame = defaultdict(set)
for user, game in dataTrain:
    gamesPerUser[user].add(game)
    usersPerGame[game].add(user)

# %%
### Question 1

# %%
# Evaluate baseline strategy

def baseline_predict(dataValid, thresh):
  popular = set()
  count = 0
  for ic, i in mostPopular:
    count += ic
    popular.add(i)
    if count > totalPlayed * thresh: break

  predictions = [0] * len(dataValid)
  for i in range(len(dataValid)):
    _, g = dataValid[i][0], dataValid[i][1]
    predictions[i] = 1 if g in popular else 0
  
  return predictions

# %%
predictions = baseline_predict(dataValid, 0.5)
accuracy = sum([x == y for x, y in zip(predictions, [z[2] for z in dataValid])]) / len(dataValid)

# %%
answers['Q1'] = accuracy

# %%
assertFloat(answers['Q1'])

# %%
### Question 2

# %%
# Improved strategy


# %%
# Evaluate baseline strategy
thresholds = [x / 10 for x in range(1, 10)]
bestPopAccuracy, bestPopThreshold = 0, None

for thresh in thresholds:
    predictions = baseline_predict(dataValid, thresh)
    accuracy = sum([x == y for x, y in zip(predictions, [z[2] for z in dataValid])]) / len(dataValid)
    if accuracy > bestPopAccuracy:
        bestPopAccuracy = accuracy
        bestPopThreshold = thresh

# %%
answers['Q2'] = [bestPopAccuracy, bestPopThreshold]

# %%
assertFloatList(answers['Q2'], 2)

# %%
### Question 3/4

# %%
def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    if denom == 0: return 0
    return numer / denom

def jaccard_predict(dataValid, thresh):
    predictions = [0] * len(dataValid)
    for i in range(len(dataValid)):
        u, g, _ = dataValid[i]
        bestSim = 0
        for g_ in gamesPerUser[u]:
            if g == g_: continue
            bestSim = max(bestSim, Jaccard(usersPerGame[g], usersPerGame[g_]))
        predictions[i] = 1 if bestSim > thresh else 0
    
    return predictions

def jaccard_pop_predict(dataValid, threshJac, threshPop):
    popular = set()
    count = 0
    for ic, i in mostPopular:
        count += ic
        popular.add(i)
        if count > totalPlayed * threshPop: break

    predictions = [0] * len(dataValid)
    for i in range(len(dataValid)):
        u, g, _ = dataValid[i]
        if g not in popular: continue
        bestSim = 0
        for g_ in gamesPerUser[u]:
            if g == g_: continue
            bestSim = max(bestSim, Jaccard(usersPerGame[g], usersPerGame[g_]))
        predictions[i] = 1 if bestSim > threshJac else 0
    
    return predictions

# %%
thresholds = [x / 100 for x in range(1, 6)]
bestJacAccuracy, bestJacThreshold = 0, None

for thresh in thresholds:
    predictions = jaccard_predict(dataValid, thresh)
    accuracy = sum([x == y for x, y in zip(predictions, [z[2] for z in dataValid])]) / len(dataValid)
    if accuracy > bestJacAccuracy:
        bestJacAccuracy = accuracy
        bestJacThreshold = thresh
print(bestJacAccuracy, bestJacThreshold)

# %%
predictions = jaccard_predict(dataValid, bestJacThreshold)
q3_accuracy = sum([x == y for x, y in zip(predictions, [z[2] for z in dataValid])]) / len(dataValid)

predictions = jaccard_pop_predict(dataValid, bestJacThreshold, bestPopThreshold)
q4_accuracy = sum([x == y for x, y in zip(predictions, [z[2] for z in dataValid])]) / len(dataValid)

answers['Q3'] = q3_accuracy
answers['Q4'] = q4_accuracy

# %%
assertFloat(answers['Q3'])
assertFloat(answers['Q4'])

# %%
### Question 5

# %%
predictions = open("predictions_Played.csv", 'w')
for l in open("pairs_Played.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u, g = l.strip().split(',')
    
    bestSim = 0
    for g_ in gamesPerUser[u]:
        if g == g_: continue
        bestSim = max(bestSim, Jaccard(usersPerGame[g], usersPerGame[g_]))
    pred = 1 if bestSim > bestJacThreshold else 0
    
    _ = predictions.write(u + ',' + g + ',' + str(pred) + '\n')

predictions.close()

# %%
answers['Q5'] = "I confirm that I have uploaded an assignment submission to gradescope"

# %%
##################################################
# Hours played prediction                        #
##################################################

# %%
hoursTrain = [(x, y, z['hours_transformed']) for (x, y, z) in allHours[:165000]]
hoursValid = [(x, y, z['hours_transformed']) for (x, y, z) in allHours[165000:]]
trainHours = [h for _, _, h in hoursTrain]
globalAverage = sum(trainHours) / len(trainHours)

# %%
hours = defaultdict()
users, items = set(), set()
usersPerItem, itemsPerUser = defaultdict(set), defaultdict(set)

for u, i, h in hoursTrain:
    hours[(u, i)] = h
    users.add(u)
    items.add(i)
    usersPerItem[i].add(u)
    itemsPerUser[u].add(i)

# %%
### Question 6

# %%
alpha = globalAverage
betaU = {u: 0 for u in users}
betaI = {i: 0 for i in items}

# %%
def iterate(alpha, betaU, betaI, lamb):
    alpha = sum([h - betaU[u] - betaI[i] for u, i, h in hoursTrain]) / len(hoursTrain)

    for u in users:
        betaU[u] = sum([hours[(u, i)] - alpha - betaI[i] for i in itemsPerUser[u]]) / (lamb + len(itemsPerUser[u]))
    
    for i in items:
        betaI[i] = sum([hours[(u, i)] - alpha - betaU[u] for u in usersPerItem[i]]) / (lamb + len(usersPerItem[i]))
    
    return alpha, betaU, betaI


# %%
currMSE = float('inf')
MSE_THRESH = 0.1
LAMB = 1

while True:
    alpha, betaU, betaI = iterate(alpha, betaU, betaI, LAMB)
    res = sum([(alpha + betaU[u] + betaI[i] - h)**2 for u, i, h in hoursTrain])
    res += LAMB * (sum([betaU[u]**2 for u in betaU]) + sum([betaI[i]**2 for i in betaI]))
    res /= len(hoursTrain)
    if currMSE - res < MSE_THRESH: break
    else: currMSE = res

# %%
validMSE = sum([(alpha + betaU[u] + betaI[i] - h)**2 for u, i, h in hoursValid]) / len(hoursValid)
answers['Q6'] = validMSE

# %%
assertFloat(answers['Q6'])

# %%
### Question 7

# %%
betaUs = [(betaU[u], u) for u in betaU]
betaIs = [(betaI[i], i) for i in betaI]
betaUs.sort()
betaIs.sort()

print("Maximum betaU = " + str(betaUs[-1][1]) + ' (' + str(betaUs[-1][0]) + ')')
print("Maximum betaI = " + str(betaIs[-1][1]) + ' (' + str(betaIs[-1][0]) + ')')
print("Minimum betaU = " + str(betaUs[0][1]) + ' (' + str(betaUs[0][0]) + ')')
print("Minimum betaI = " + str(betaIs[0][1]) + ' (' + str(betaIs[0][0]) + ')')

# %%
answers['Q7'] = [betaUs[-1][0], betaUs[0][0], betaIs[-1][0], betaIs[0][0]]

# %%
answers['Q7']

# %%
assertFloatList(answers['Q7'], 4)

# %%
### Question 8

# %%
# Better lambda...
lambdas = [0.0001, 0.001, 0.01, 0.1, 1, 10] 
bestLambda, bestMSE = None, float('inf')

for lamb in lambdas:
    currMSE = float('inf')
    while True:
        alpha, betaU, betaI = iterate(alpha, betaU, betaI, lamb)
        res = sum([(alpha + betaU[u] + betaI[i] - h)**2 for u, i, h in hoursTrain])
        res += lamb * (sum([betaU[u]**2 for u in betaU]) + sum([betaI[i]**2 for i in betaI]))
        res /= len(hoursTrain)
        if currMSE - res < MSE_THRESH: break
        else: currMSE = res
    if currMSE < bestMSE:
        bestMSE = currMSE
        bestLambda = lamb

# %%
answers['Q8'] = (bestLambda, bestMSE)

# %%
assertFloatList(answers['Q8'], 2)

# %%
LAMB = bestLambda
currMSE = float('inf')
while True:
    alpha, betaU, betaI = iterate(alpha, betaU, betaI, lamb)
    res = sum([(alpha + betaU[u] + betaI[i] - h)**2 for u, i, h in hoursTrain])
    res += lamb * (sum([betaU[u]**2 for u in betaU]) + sum([betaI[i]**2 for i in betaI]))
    res /= len(hoursTrain)
    if currMSE - res < MSE_THRESH: break
    else: currMSE = res

# %%
predictions = open("predictions_Hours.csv", 'w')
for l in open("pairs_Hours.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,g = l.strip().split(',')
    
    bu = betaU[u] if u in betaU else 0
    bi = betaI[g] if g in betaI else 0
    
    _ = predictions.write(u + ',' + g + ',' + str(alpha + bu + bi) + '\n')

predictions.close()

# %%
f = open("answers_hw3.txt", 'w')
f.write(str(answers) + '\n')
f.close()

# %%



