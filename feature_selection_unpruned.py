import pandas as pd
import numpy as np
import time
import sys

instances = 0
numfeatures = 0


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # Ensures immediate writing

    def flush(self):
        for f in self.files:
            f.flush()

def train_and_test(df: pd.DataFrame, features: list):

    global instances

    if(len(features) ==0):
        X = np.zeros((instances, 1))
    else:
        X = df.iloc[:, features].to_numpy()

    y = df.iloc[:, 0].to_numpy()
    correctPredictions = 0

    for i in range (instances):
        diffs = X-X[i] #calculate the differences between datapoints
        distances = np.einsum('ij,ij->i',diffs,diffs) #calculate the squares of the distances rather than the distaces because the square-root funtion adds time and overhead
        distances[i] = np.inf #exclude itself
        nn = np.argmin(distances) #find the closest datapoint

        if(y[i] == y[nn]):
            correctPredictions+=1
    
    return correctPredictions/instances

def forward_search(df: pd.DataFrame):
    features = list()
    max_accuracies = list()
    print("On iteration 0:")
    features_to_test = list()
    accuracy = train_and_test(df, features_to_test)
    print(f"\tTesting no features with accuracy {(100*accuracy):.2f}%. Features tested: {features_to_test}")
    max_accuracies.append(accuracy)

    global numfeatures

    for i in range(1, numfeatures+1):
        print(f"On iteration {i}")
        max_accuracy = 0
        max_accuracy_feature = 0

        for k in range(1, numfeatures+1):
            if(k not in features):
                accuracy = 0
                features_to_test = list()
                for feature in features:
                    features_to_test.append(feature)
                features_to_test.append(k)
                accuracy = train_and_test(df, features_to_test)
                print(f"\tTesting adding feature {k} with accuracy {(100*accuracy):.2f}%. Features tested: {features_to_test}")
                if(accuracy>max_accuracy):
                    max_accuracy = accuracy
                    max_accuracy_feature = k

        features.append(max_accuracy_feature)
        max_accuracies.append(max_accuracy)
        print(f"Feature set {features} was the best, with accuracy {(100*max_accuracy):.2f}%")
        if(max_accuracy < max_accuracies[len(max_accuracies)-2]):
            print("Warning! Accuracy has decreased.")

    return (features, max_accuracies)

def backward_search(df: pd.DataFrame):
    features = list()
    for i in range(1, df.shape[1]):
        features.append(i)
    features_removed = list()
    max_accuracies = list()
    global numfeatures

    for i in range(0, numfeatures+1):
        print(f"On iteration {i}")
        max_accuracy = 0
        max_accuracy_feature_to_remove = 0
        
        if(i ==0):
            features_to_test = list()
            for feature in features:
                features_to_test.append(feature)
            accuracy = train_and_test(df, features_to_test)
            print(f"\tTesting all features with accuracy {(100*accuracy):.2f}%. Features tested: {features_to_test}")
            max_accuracies.append(accuracy)
            if(accuracy>max_accuracy):
                max_accuracy = accuracy
            continue

        for k in range(1, numfeatures+1):
            if(k in features):
                accuracy = 0
                features_to_test = list()
                for feature in features:
                    features_to_test.append(feature)
                features_to_test.remove(k)
                accuracy = train_and_test(df, features_to_test)
                print(f"\tTesting removing feature {k} with accuracy {(100*accuracy):.2f}%. Features tested: {features_to_test}")
                if(accuracy>max_accuracy):
                    max_accuracy = accuracy
                    max_accuracy_feature_to_remove = k

        features.remove(max_accuracy_feature_to_remove)
        features_removed.append(max_accuracy_feature_to_remove)
        max_accuracies.append(max_accuracy)
        print(f"Feature set {features} was the best, with accuracy {(100*max_accuracy):.2f}%")
        if(max_accuracy < max_accuracies[len(max_accuracies)-2]):
            print("Warning! Accuracy has decreased.")

    return (features_removed, max_accuracies)

def main():
    log_file = open('output.txt', 'w')
    sys.stdout = Tee(sys.stdout, log_file)
    filePath = input("Please enter the filename that contains the data:")
    df = pd.read_csv(filePath, sep='\s+', header=None, index_col=False)
    print("DataFrame read from " + filePath)
    alg = int(input("Enter 1 for forward selection and 2 for backward elimination:\n"))
    global instances
    global numfeatures
    instances = df.shape[0]
    numfeatures = df.shape[1]-1
    print(f"The dataset has {numfeatures} features, not including class atribute, and {instances} instances.\nBegining Search:")
    if alg == 1:
        start_time = time.perf_counter()
        (features_added, max_accuracies) = forward_search(df)
        end_time = time.perf_counter()

        max_accuracy = 0
        max_index = 0
        results = pd.DataFrame(index=range(len(max_accuracies)), columns=range(2))
        for i in range (len(max_accuracies)):
            accuracy = max_accuracies[i]
            results.iloc[i, 0] = features_added[:i]
            results.iloc[i,1] = accuracy
            if(accuracy>max_accuracy):
                max_accuracy =accuracy
                max_index = i

        features = features_added[:max_index]
        print(f"Finished search. Greatest accuracy = {(max_accuracy*100):.2f}% found with features {features}")
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")

        results.to_csv("results.csv", index=False)

    elif alg ==2:
        start_time = time.perf_counter()
        (features_removed,max_accuracies) = backward_search(df)
        end_time = time.perf_counter()
        max_accuracy = 0
        max_index = 0
        results = pd.DataFrame(index=range(len(max_accuracies)), columns=range(2))
        for i in range (len(max_accuracies)):
            accuracy = max_accuracies[i]
            results.iloc[i, 0] = features_removed[i:]
            results.iloc[i,1] = accuracy
            if(accuracy>max_accuracy):
                max_accuracy = accuracy
                max_index = i
        
        features = features_removed[max_index:]
        print(f"Finished search. Greatest accuracy = {(max_accuracy*100):.2f}% found with features {features}")

        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")

        results.to_csv("results.csv", index=False)

    log_file.close()


if __name__ == "__main__":
    main()