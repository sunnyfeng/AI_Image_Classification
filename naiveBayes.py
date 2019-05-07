# naiveBayes.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import util
import classificationMethod
import math

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
  """
  See the project description for the specifications of the Naive Bayes classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__(self, legalLabels):
    self.legalLabels = legalLabels
    self.type = "naivebayes"
    self.k = 1 # this is the smoothing parameter, ** use it in your train method **
    self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **
    
  def setSmoothing(self, k):
    """
    This is used by the main method to change the smoothing parameter before training.
    Do not modify this method.
    """
    self.k = k

  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    """
    Outside shell to call your method. Do not modify this method.
    """  
      
    # might be useful in your code later...
    # this is a list of all features in the training set.
    self.features = list(set([ f for datum in trainingData for f in datum.keys() ]));
    
    if (self.automaticTuning):
        kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
    else:
        kgrid = [self.k]
        
    self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)
      
  def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
    """
    Trains the classifier by collecting counts over the training data, and
    stores the Laplace smoothed estimates so that they can be used to classify.
    Evaluate each value of k in kgrid to choose the smoothing parameter 
    that gives the best accuracy on the held-out validationData.
    
    trainingData and validationData are lists of feature Counters.  The corresponding
    label lists contain the correct label for each datum.
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """


    normLabel = util.Counter() 
    normFeatProb = util.Counter() 
    normCounts = util.Counter() 
    
    sizeTD = len(trainingData)   

    for i in range(sizeTD):
        dataPoint = trainingData[i]
        label = trainingLabels[i]
        normLabel[label] = normLabel[label] + 1
        for feat, value in dataPoint.items():
            normCounts[(feat,label)] = normCounts[(feat,label)] + 1
            if value > 0: 
                normFeatProb[(feat, label)] = normFeatProb[(feat, label)] + 1

    for k in kgrid: 
    
        counts = util.Counter()
        for key, val in normCounts.items():
            counts[key] = counts[key] + val

        past = util.Counter()
        for key, val in normLabel.items():
            past[key] = past[key] + val

        featProb = util.Counter()
        for key, val in normFeatProb.items():
            featProb[key] = featProb[key] + val

        smoothingFactor = 2.4*k

        for label in self.legalLabels:
            for feat in self.features:
                featProb[ (feat, label)] =  featProb[ (feat, label)] + k
                counts[(feat, label)] = counts[(feat, label)] + smoothingFactor 

        
        past.normalize()

        for f, count in featProb.items():
            featProb[f] = float(count) / counts[f]

        self.featProb = featProb
        self.past = past

        predictions = self.classify(validationData)
        accuracyCount =  [predictions[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)
        
        highestAcc = -1  

        if accuracyCount > highestAcc:
            bestParams = (past, featProb, k)
            highestAcc = accuracyCount
    

    self.past, self.featProb, self.k = bestParams



        
  def classify(self, testData):
    """
    Classify the data based on the posterior distribution over labels.
    
    You shouldn't modify this method.
    """
    guesses = []
    self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
    for datum in testData:
        posterior = self.calculateLogJointProbabilities(datum)
        guesses.append(posterior.argMax())
        self.posteriors.append(posterior)
    return guesses
      
  def calculateLogJointProbabilities(self, datum):
    """
    Returns the log-joint distribution over legal labels and the datum.
    Each log-probability should be stored in the log-joint counter, e.g.    
    logJoint[3] = <Estimate of log( P(Label = 3, datum) )>
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """
    lJ = util.Counter()
    for label in self.legalLabels:
        lJ[label] = math.log(self.past[label])
        for feat, value in datum.items():
            if value > 0:
                lJ[label] = lJ[label] + math.log(self.featProb[feat,label])
            if value < 0:
                lJ[label] =  lJ[label] + math.log(1-self.featProb[feat,label])
            if value == 0:
                lJ[label] =  lJ[label] + math.log(1-self.featProb[feat,label])
    return lJ
  
  def findHighOddsFeatures(self, label1, label2):
    """
    Returns the 100 best features for the odds ratio:
            P(feature=1 | label1)/P(feature=1 | label2) 
    
    Note: you may find 'self.features' a useful way to loop through all possible features
    """
    for fs in self.features:
        val = (self.featProb[feat, first]/self.featProb[fs, second], fs)
        allFOs.append(val)
    allFOs.sort()
    allFOs = [fs for val, feat in allFOs[-100:]]

    return allFOs
    

    
      
