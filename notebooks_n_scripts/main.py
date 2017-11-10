from AlgoTestFactory import *
import os


def printAccuracies(path):
    print(path)
    fNames = [validatePickleName(f) for f in os.listdir(path)]
    fNames = list(filter(len, fNames))
    accuracies = []
    for fName in fNames:
        act = pickle.load(open(path + fName, 'rb'))
        for f, pair in act.items():
            accuracies.append(pair[0]*100/pair[1])

    print(len(accuracies), 'files processed')
    accuracies = np.array(accuracies)
    print("Results: ")
    print("Mean: ", accuracies.mean())
    print("Percentiles: ", np.percentile(accuracies, [0, 25, 50, 75, 100]))
    print('Macro Accuracy %:', 100*np.sum(accuracies >= 95)/accuracies.shape[0])

if __name__ == "__main__":
    path = "3RWR_Unsup_QNEW_66"
    print('Modules Loaded... ID-', path)

    # 3RWR or 25Path
    altf1 = AlgoTestFactory([50000, 100078], 12, savePath = path, storeAccuracies=True, partition = [0.33, 0.33, 0.33, 1], algoname = '3RWR')
    altf1.run()
    printAccuracies('.temp/%s/' % path)
