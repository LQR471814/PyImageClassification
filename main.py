import numpy as np
import math
import plotly.express as px
import gzip
import os

class NeuralNetwork:

    def __init__(self, layerSize:list):

        self.weights1 = np.array(self.generateDataList([layerSize[0], layerSize[1]], 1))
        self.weights2 = np.array(self.generateDataList([layerSize[1], layerSize[2]], 1))
        self.biases1 = np.array(self.generateDataList([1, layerSize[1]], 0))
        self.biases2 = np.array(self.generateDataList([1, layerSize[2]], 0))

        print("Checking Weights and Biases! =============================================>")
        # self.prettyPrintList(self.weights1, "weights1")
        # self.prettyPrintList(self.weights2, "weights2")
        # self.prettyPrintList(self.biases1, "biases1")
        # self.prettyPrintList(self.biases2, "biases2")

    def predict(self, inputValues):
        print("Prediction =============================================>")
        # Set1 = [[0, 0], [0, 1], [1, 0], [1, 1]]
        # Set2 = [[9, 9], [10, 9], [9, 10], [10, 10]]
        # self.totalSet = [Set1, Set2]

        # weights1 = [[0, 0], [0, 0], [0, 0]]
        # weights2 = [[0, 0, 0], [0, 0, 0]]
        # bias1 = [[0], [0], [0]]
        # bias2 = [[0], [0], [0]]

        self.inputValues = np.array(self.transformInput(inputValues))

        # self.prettyPrintList(self.inputValues, "inputValues")

        # biasValues1 = np.array(bias1)
        # biasValues2 = np.array(bias2)
        # weightMatrix1 = np.array(weights1)
        # weightMatrix2 = np.array(weights2)

        # hiddenLayer1 = weightMatrix1.dot(inputValues) + biasValues
        hiddenLayer1 = np.matmul(self.weights1, self.inputValues) + self.biases1

        self.a1 = self.activationFunction(hiddenLayer1)
        self.a1 = np.array(self.a1)

        # self.prettyPrintList(self.a1, "a1")

        hiddenLayer2 = np.matmul(self.weights2, self.a1) + self.biases2

        self.a2 = self.activationFunction(hiddenLayer2)
        self.a2 = np.array(self.a2)

        self.prettyPrintList(self.a2, "a2")

        # graph = px.scatter(x=self.generateGraphListFromDataSet(self.totalSet)[0], y=self.generateGraphListFromDataSet(self.totalSet)[1])
        # graph.show()

    def generateOutput(self, numbOutputs:int) -> np.array:
        output = []

        for i in range(numbOutputs):
            output.append(1)


    def transformInput(self, inputVals:list) -> list:
        returnList = []

        for val in inputVals:
            returnList.append([val])

        return returnList

    def generateGraphListFromDataSet(self, dataSet:list) -> list:
        xVals = []
        yVals = []

        for data in dataSet:
            for coord in data:
                xVals.append(coord[0])
                yVals.append(coord[1])

        return [xVals, yVals]

    def activationFunction(self, x:int):
        return 1 / (1 + (math.exp(1) ** -x))

    def generateDataList(self, layerSize:list, randomIncrement:int) -> list:
        #? The layerSize is in the format of [items per row, number rows]

        returnList = []

        for r in range(layerSize[1]):
            tempList = []
            for i in range(layerSize[0]):
                #? So apparently you would use the standard distribution divided by the squareroot
                #?    of the number of inputs to get good results even if you have a large amount of inputs.
                #? Although I'm not sure why you would use the squareroot of the number of inputs.
                tempList.append((np.random.standard_normal() / layerSize[0]**0.5) * randomIncrement)
            returnList.append(tempList)

        return returnList

    def prettyPrintList(self, inpList:list, name:str) -> None:
        print("------------------------------------> " + name)
        for lst in inpList:
           print(lst)

def createInput(length:int, val:int) -> list:
    returnList = []
    for i in range(length):
        returnList.append(val)
    return returnList

#? Training Data
image_size = 28
num_images = 100

pathToTrainingData = os.path.dirname(os.path.abspath(__file__)) + "\\" + "train-images-idx3-ubyte.gz"
trainingData = gzip.open(pathToTrainingData, "r")
trainingData.read(16)
buf = trainingData.read(image_size * image_size * num_images)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
data = data.reshape(num_images, image_size, image_size, 1)
imageArray = np.asarray(data[1]).squeeze()

def twoDimArrayToOneDimensionalList(array):
    returnArray = []
    for line in array:
        for point in line:
            returnArray.append(point)

    return returnArray

#? Neural Network Settings
inputNumb = image_size ** 2
neuronNumb = 10
outputNumb = 10
randomIncrement = 1

neuralNetObj = NeuralNetwork([inputNumb, neuronNumb, outputNumb])
neuralNetObj.predict(twoDimArrayToOneDimensionalList(imageArray))