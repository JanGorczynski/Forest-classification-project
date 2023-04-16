import numpy as np
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import pickle
import dill
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout,InputLayer, BatchNormalization,Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

#Reading data and spliting it
df = pd.read_csv("data\covtype.data",header=None)
X = df.loc[:,:53]
y = df[54]
#This is a three way split into training, testing and validation data. Testing will be used for choosing good hyperparmeters, validation for finall model  assessment
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.5, random_state=42)
#This is heuristic model working by simple principle, data has 7 different classes, 
#so we save in list "coef" average values of every feature for each class divided by maximal value of that feature.
#When predicting we find minimal squared error of all features of the instance divided by max value and features of all classes saved in the list coef
#then we find the class which has the most similar features to our instance and predict that class
class HeuristicModel:
    def __init__(self):
        self.coef = []
        self.max = []
    def fit(self,X_train,y_train):
        for i in range(0,53):
            self.max.append(X_train[i].max())
        
        for i in range(1,8):
            coef_list = []
            for column in range(0,53):
                coef_list.append(X_train[y_train == i][column].mean()/self.max[column])
            self.coef.append(coef_list)
    def predict_one(self,X_instance):
        diff_list = []
        for i in range(0,7):
            diff = 0
            for j in range(0,53):
                diff += (X_instance[j]/self.max[j] - self.coef[i][j])**2
            diff_list.append(diff)
        return  diff_list.index(min(diff_list)) + 1
    def predict(self,X_test):
        X_test = pd.DataFrame(X_test)
        predictions = []
        for i in X_test.index:
            predictions.append(self.predict_one(X_test.loc[i]))
        return predictions


with open('saved_models/h_class.pkl', 'wb') as f:
    dill.dump(HeuristicModel, f)


print("---Now training and testing heuristic model this may take a while---")
heuristicModel = HeuristicModel()
#Fitting model to data
heuristicModel.fit(X_train,y_train)
#Saving model in predifined file
with open('saved_models\heuristic.pkl', 'wb') as f:
    pickle.dump(heuristicModel, f)
#Making predictions and measuring accuracy
heuristicPred = heuristicModel.predict(X_test)
heuristicAccuracy = len(y_test[y_test == heuristicPred])/len(y_test)
print("accuracy: ",heuristicAccuracy)
print("\n\n")


#For the rest of the models we will use min max scaler to normalize the data
scaler = StandardScaler()
scaler.fit(X_train)
#Saving scaler
with open('saved_models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
#Normalizing data
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)

#First algorithm: k-nearest neighbors
print("---Now training and testing  k-nearest neighbors model this may take a while---")
KnnModel = KNeighborsClassifier(n_neighbors=5)
KnnModel.fit(X_train,y_train)
with open('saved_models/knn_model.pkl', 'wb') as f:
    pickle.dump(KnnModel, f)
KnnPred = KnnModel.predict(X_test)
KnnAccuracy = len(y_test[y_test == KnnPred])/len(y_test)
print("accuracy: ",KnnAccuracy)
print("\n\n")

#Second algorithm: decision tree
print("---Now training and testing decision tree model---")
DTreeModel = DecisionTreeClassifier()
DTreeModel.fit(X_train,y_train)
with open('saved_models/dtree_model.pkl', 'wb') as f:
    pickle.dump(KnnModel, f)
DTreePred = DTreeModel.predict(X_test)
DTreeAccuracy = len(y_test[y_test == DTreePred])/len(y_test)
print("accuracy: ",DTreeAccuracy)
print("\n\n")
#Neural Networks

#Changing y values to categorical to fit in neural network
y_train_categorical = to_categorical([i-1 for i in y_train])
y_test_catogorical = to_categorical([i-1 for i in y_test])

#Two function returning neural network structure, we will use those function to find best achitecture and hyperparameters

#Deeper structure
def model_structure_1():
    model = Sequential()
    model.add(InputLayer(54))

    model.add(Dense(128,activation = "relu"))
    model.add(BatchNormalization())

    model.add(Dense(128,activation = "relu"))
    model.add(BatchNormalization())
    
    model.add(Dense(128,activation = "relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(128,activation = "relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(7, activation='softmax'))
    return model 

#Wider structure
def model_structure_2():
    model = Sequential()
    model.add(InputLayer(54))

    model.add(Dense(128,activation = "relu"))
    model.add(BatchNormalization())

    model.add(Dense(256,activation = "relu"))
    model.add(BatchNormalization())

    model.add(Dense(256,activation = "relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(7, activation='softmax'))
    return model

#List of data needed for neural network training and testing
data = [X_train,y_train_categorical,X_test,y_test_catogorical]

#Function that checks all possible combinations of structure, learning rate and batch size then returns all the models, their accuracy,
#their history and hyperparameters
def gridSearch(model_structures,lr,b_size,data):
    models = []
    X_train = data[0]
    y_train = data[1]
    X_test = data[2]
    y_test = data[3]
    for model_structure in model_structures:
        for rate in lr:
            for size in b_size:
                hiperparamiters = "Structure: " + model_structure.__name__ + " learning rate: " + str(rate) + " batch size: " + str(size)
                print(hiperparamiters)
                model = model_structure()
                model.compile(loss= "CategoricalCrossentropy",optimizer = Adam(learning_rate=rate), metrics=['accuracy'])
                history = model.fit(X_train, y_train, batch_size=size, epochs=20, verbose=0,validation_data=(X_test, y_test))
                accuracy = model.evaluate(X_test, y_test_catogorical)[1]
                models.append({"accuracy": accuracy,"model":model,"history":history,
                              "hyperparamiters":hiperparamiters})
    return models


#Using grid Search function to find good hyperparameters
print("--Training neural nets, this might take a while---")
models = gridSearch([model_structure_1,model_structure_2],[0.01,0.001],[512,1024],data)
print("\n\n")

#Sorting models by accuracy and finding the best one
models = sorted(models, key=lambda x: -x["accuracy"])
best_model = models[0]
print("The best accuracy: ",best_model["accuracy"]," for hyperparameters: ",best_model["hyperparamiters"]  )

#Ploting training curves for best model
hist = best_model["history"]
plt.plot(hist.history["loss"],label="x")
plt.plot(hist.history["val_loss"])
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(['Training loss', 'Validation loss'])
plt.show()

#Ploting accuracy for different algorithms
accuracy = pd.DataFrame({"Heuristics": [heuristicAccuracy],"Knn": [KnnAccuracy],"Decision tree":[DTreeAccuracy],"Neural network":
                        [best_model["accuracy"]]})
accuracy = accuracy.transpose()
accuracy.columns = ["accuracy"]
plt.bar(x=accuracy.index, height=accuracy['accuracy'], width=0.9)
plt.ylabel("accuracy")
plt.show()

#Making predictions using our model
NNmodel = best_model["model"]
NNmodel.save('saved_models/neural_network.h5')
NNpred = NNmodel.predict(X_test)
NNpred = [np.argmax(i) + 1 for i in NNpred]

#Evaluating all algorithms using confusion matrix and classification report
print("\n\n")
print("Heuristics")
print("Confusion matrix")
print(confusion_matrix(y_test,heuristicPred))
print("Classification report")
print(classification_report(y_test,heuristicPred))
print("\n\n")

print("KNN clasifier")
print("Confusion matrix")
print(confusion_matrix(y_test,KnnPred))
print("Classification report")
print(classification_report(y_test,KnnPred))
print("\n\n")

print("Decisision tree clasifier")
print("Confusion matrix")
print(confusion_matrix(y_test,DTreePred))
print("Classification report")
print(classification_report(y_test,DTreePred))
print("\n\n")

print("Neural network")
print("Confusion matrix")
print(confusion_matrix(y_test,NNpred))
print("Classification report")
print(classification_report(y_test,NNpred))
print("\n\n")

#Now we'll use validation data to assess our models, we'll skip heuristic algortihm since it takes long time to make predictions
print("---Now testing algorithms on validation data to get final accuracy---")

pred = KnnModel.predict(X_val)
accuracy = len(y_val[y_val == pred])/len(y_val)
print("KNN acuracy = ",accuracy)

pred = DTreeModel.predict(X_val)
accuracy = len(y_val[y_val == pred])/len(y_val)
print("Decision tree acuracy = ",accuracy)

pred = NNmodel.predict(X_val)
pred = [np.argmax(i) + 1 for i in pred]
accuracy = len(y_val[y_val == pred])/len(y_val)
print("Neural network acuracy = ",accuracy)

