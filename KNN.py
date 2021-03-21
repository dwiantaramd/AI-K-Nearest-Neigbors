import csv
import math
import pandas as pd

# Load a CSV file
def load_csv():
    dataset = list()
    with open('Diabetes.csv', 'r') as file:
        data = csv.reader(file)
        next(data, None)
        for row in data:
            dataset.append(row)
       
    for column in range(len(dataset[0])):
        for row in dataset:
            row[column] = float(row[column])
    return dataset

#Function to split dataset into folds
def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

#Numeric Distance using euclidean distance
def euclidean_distance(x1, x2):
    distance = 0
    for i in range(len(x1)-1):
        distance += (x1[i] - x2[i])**2
    return math.sqrt(distance)

#Function to scale dataset between [0, 1]
def normalization(dataset):
    minmax = list()
    for i in range(len(dataset[0])-1):
        column = list()
        for row in dataset:
            column.append(row[i])
        minimum = min(column)
        maximum = max(column)
        minmax.append([minimum, maximum])
        
    for row in dataset:
        for i in range(len(row)-1):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

#Function to return k closest neighbor
def get_neighbors(train, test_row, k):
    distances = list()
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
        
    distances.sort(key=lambda x: x[1])
    neighbors = list()
    for i in range(k):
        neighbors.append(distances[i][0])
    return neighbors

#Function to return testing row class prediction/estimation
def classification(train, test_row, k):
    neighbors = get_neighbors(train, test_row, k)
    outcome = [row[-1] for row in neighbors]
    prediction = max(set(outcome), key=outcome.count)
    return prediction

#Function to return testing set class prediction/estimation
def k_nearest_neighbors(data_test, data_train, k):
    estimation = list()
    for row in data_test:
        temp = classification(data_train, row, k)
        estimation.append(temp)
    return estimation

#Function to count accuracy of testing set prediction
def accuracy(testing_set_outcome, prediction):
    correct = 0
    for i in range(len(testing_set_outcome)):
        if testing_set_outcome[i] == prediction[i]:
            correct += 1
    return correct / float(len(testing_set_outcome)) * 100.0

#Main Program
dataset = load_csv()
normalization(dataset)

#5-Fold-Cross Validation
five_fold = list(split(dataset, 5))
acc_mean_list = list()
for k in range(1,31):
    acc = list()
    n_neighbors = k
    for fold in five_fold:
        training_set = list(five_fold)
        training_set.remove(fold)
        training_set = sum(training_set, [])
        testing_set = list()
        for row in fold:
            row_copy = list(row)
            testing_set.append(row_copy)
            row_copy[-1] = None
        predicted = k_nearest_neighbors(testing_set, training_set, n_neighbors)
        outcome = [row[-1] for row in fold]
        accs = accuracy(outcome, predicted)
        acc.append(accs)
    mean = sum(acc)/float(5)
    acc_mean_list.append([mean,n_neighbors])
    print("K : ", n_neighbors)
    print("Accuracy Mean : ", mean)
    print("==================================")


df = pd.DataFrame(acc_mean_list, columns=['Accuracy Mean', 'K'])
ax2 = df.plot.scatter(x='K', y='Accuracy Mean')
print("")
print("Best K Value : " , max(acc_mean_list,key=lambda x:x[0])[1])
print("Accuracy Mean : ", max(acc_mean_list,key=lambda x:x[0])[0])