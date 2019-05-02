# kNearest.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import util
import classificationMethod
import numpy
import operator
import scipy
from scipy.spatial import distance

class kNearestClassifier(classificationMethod.ClassificationMethod):
  """
  The MostFrequentClassifier is a very simple classifier: for
  every test instance presented to it, the classifier returns
  the label that was seen most often in the training data.
  """
  def __init__(self, legalLabels):
    self.guess = None
    self.type = "kNearest"
    self.k = 5
  
  def train(self, data, labels, validationData, validationLabels):
    """
    Find the most common label in the training data.
    """
    #counter = util.Counter()
    #counter.incrementAll(labels, 1)
    #self.guess = counter.argMax()
    neigh = self.getNeighbors(data, labels)
    self.predict(neigh)
  
  def classify(self, data):
    """
    Classify all test data as the most common label.
    """
    #return [self.guess for i in testData]
    guesses = []
    for datum in data:
      vectors = util.Counter()
      for l in self.legalLabels:
        vectors[l] = self.weights[l] * datum
      guesses.append(vectors.argMax())
    return guesses


  def euclideanDist(self,a,b):
    #dist = numpy.linalg.norm(a - b)
    #return dist
    return distance.euclidean(a, b)

  def getNeighbors(self, data, labels):
    dists = []
    for x in range(len(data)):
        dArr = []
        dArr = numpy.array(list(data))
        lArr = []
        lArr = numpy.array(list(labels))

        print(dArr)
        print(lArr)

        dist = self.euclideanDist(dArr,lArr)
        dists.append((data[x], dist))
    dists.sort(key=operator.itemgetter(1))
    neigh = []
    for x in range(self.k):
      neigh.append(dists[x][0])
    return neigh

  def predict(self, neigh):
    majority = {}
    for x in range(len(neigh)):
      pred = neigh[x][-1]
      if pred in majority:
        majority[pred] += 1
      else:
        majority[pred] = 1
    sortedVotes = sorted(majority.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]
