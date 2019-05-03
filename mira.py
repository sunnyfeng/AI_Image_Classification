# mira.py
# -------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# Mira implementation
import util
PRINT = True

class MiraClassifier:
  """
  Mira classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__( self, legalLabels, max_iterations):
    self.legalLabels = legalLabels
    self.type = "mira"
    self.automaticTuning = False 
    self.C = 0.001
    self.legalLabels = legalLabels
    self.max_iterations = max_iterations
    self.initializeWeightsToZero()

  def initializeWeightsToZero(self):
    "Resets the weights of each label to zero vectors" 
    self.weights = {}
    for label in self.legalLabels:
      self.weights[label] = util.Counter() # this is the data-structure you should use
  
  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    "Outside shell to call your method. Do not modify this method."  
      
    self.features = trainingData[0].keys() # this could be useful for your code later...
    
    if (self.automaticTuning):
        Cgrid = [0.002, 0.004, 0.008]
    else:
        Cgrid = [self.C]
        
    return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, Cgrid)

  def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, Cgrid):
    """
    This method sets self.weights using MIRA.  Train the classifier for each value of C in Cgrid, 
    then store the weights that give the best accuracy on the validationData.
    
    Use the provided self.weights[label] data structure so that 
    the classify method works correctly. Also, recall that a
    datum is a counter from features to values for those features
    representing a vector of values.
    """
    tempWeights = self.weights.copy()
    bestScore = 0
    bestC = 0
    for C in Cgrid:
      self.weights = tempWeights
      score = self.trainC(trainingData, trainingLabels, validationData, validationLabels, C)
      if score > bestScore:
          bestScore = score
          bestC = C
    print(bestC)
    self.weights = tempWeights
    score = self.trainC(trainingData, trainingLabels, validationData, validationLabels, bestC)


  def trainC(self, trainingData, trainingLabels, validationData, validationLabels, C):
    """
    This method sets self.weights using MIRA.  Train the classifier for each value of C in Cgrid,
    then store the weights that give the best accuracy on the validationData.

    Use the provided self.weights[label] data structure so that
    the classify method works correctly. Also, recall that a
    datum is a counter from features to values for those features
    representing a vector of values.
    """
    for iteration in range(self.max_iterations):
      print "Starting iteration ", iteration, "..."
      for i in range(len(trainingData)):
        "*** YOUR CODE HERE ***"

        data = trainingData[i]
        label = trainingLabels[i]
        pred = self.classify([data])[0]
        if pred != label:
           # numer = self.abs(tempWeights[pred] - tempWeights[label])*data + 1.0
           # denom = self.weighted(data, 2.0) * self.abs(data)
            # tau = min(Cgrid[C], 0, (numer*1.0/denom*1.0))
          tau = C
            # tempWeights[label] += self.weighted(data, tau)
            # tempWeights[pred] -= self.weighted(data, tau)
          self.weights[label] = self.weights[label] + self.weighted(data, tau)
          self.weights[pred] = self.weights[pred] - self.weighted(data, tau)

    predictions = self.classifyWithWeight(validationData,self.weights)
    correct = 0.0
    for i in range(len(predictions)):
       if predictions[i] == validationLabels[i]:
         correct += 1.0
    score = correct / (len(predictions) * 1.0)
    print(score)
    return score


  def weighted(self,data, multiple):
    new = util.Counter()
    for i in data:
      new[i] = data[i]*multiple
    return new

  def abs(self, data):
    new = util.Counter()
    for i in data:
      new[i] = abs(data[i])
    return new

  def classify(self, data):
    """
    Classifies each datum as the label that most closely matches the prototype vector
    for that label.  See the project description for details.
    
    Recall that a datum is a util.counter... 
    """
    guesses = []
    for datum in data:
      vectors = util.Counter()
      for l in self.legalLabels:
        vectors[l] = self.weights[l] * datum
      guesses.append(vectors.argMax())
    return guesses

  def classifyWithWeight(self, data, weights):
    """
    Classifies each datum as the label that most closely matches the prototype vector
    for that label.  See the project description for details.

    Recall that a datum is a util.counter...
    """
    guesses = []
    for datum in data:
      vectors = util.Counter()
      for l in self.legalLabels:
        vectors[l] = weights[l] * datum
      guesses.append(vectors.argMax())
    return guesses

  
  def findHighOddsFeatures(self, label1, label2):
    """
    Returns a list of the 100 features with the greatest difference in feature values
                     w_label1 - w_label2

    """
    featuresOdds = []
    "*** YOUR CODE HERE ***"
    #featuresOdds = (self.weights[label1]-self.weights[label2]).sortedKeys()[:100]  # gets first 100 highest weights
    return featuresOdds

