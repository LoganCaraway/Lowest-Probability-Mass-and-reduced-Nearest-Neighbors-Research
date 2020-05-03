from csv import reader
import sys
import random
import MathAndStats as ms
import NearestNeighbor as nn


def openFile(data_file):
    lines = open(data_file, "r").readlines()
    csv_lines = reader(lines)
    data = []

    for line in csv_lines:
        tmp = []
        if sys.argv[2] == 'rm':
            # remove line number from each example (first column)
            for c in range(1, len(line) - 1):
                tmp.append(float(line[c]))
        else:
            for c in range(len(line) - 1):
                tmp.append(float(line[c]))
        if sys.argv[3] == 'r':
            tmp.append(float(line[-1]))
        else:
            tmp.append(line[-1])
        data.append(tmp)

    data = ms.normalize(data)
    # divide data into 10 chunks for use in 10-fold cross validation paired t test
    chnks = getNChunks(data, 10)
    class_list = getClasses(data)
    return chnks, class_list

# divide the example set into n random chunks of approximately equal size
def getNChunks(data, n):
    # randomly shuffle the order of examples in the data set
    random.shuffle(data)
    dataLen = len(data)
    chunkLen = int(dataLen / n)
    # chunks is a list of groups of examples
    chunks = []
    # rows are observation
    # columns are labels

    # group the examples in data into chunks
    for obs in range(0, dataLen, chunkLen):
        if (obs + chunkLen) <= dataLen:
            chunk = data[obs:obs + chunkLen]
            chunks.append(chunk)
    # append the extra examples randomly to the chunks
    for i in range(n*chunkLen, dataLen):
        chunks[random.randint(0,n-1)].append(data[i])
    for i in range(len(chunks)):
        print("Length of chunk: ", len(chunks[i]))
    return chunks

def getClasses(data):
    if sys.argv[3] == 'r':
        return []
    classes = []
    for obs in range(len(data)):
        if not data[obs][-1] in classes:
            classes.append(data[obs][-1])
    return classes

def tenFoldCV(chunked_data, clss_list, use_regression, k, k2, t, phi, kc, kc2, ct, cphi,
              ke, ke2, et, ephi):
    knn_missed = []
    knn2_missed = []
    cnn_missed = []
    cnn2_missed = []
    enn_missed = []
    enn2_missed = []
    for test_num in range(10):
        print("\n\n\nFold: ", test_num+1, "of 10 fold cross validation")
        training_set = []

        testing_set = chunked_data[test_num]
        # make example set
        for train in range(10):
            if train != test_num:
                for x in range(len(chunked_data[train])):
                    training_set.append(chunked_data[train][x])

        validation_index = int((float(len(training_set)) * 8 / 10)) - 1
        knn = nn.NearestNeighbor(training_set, k)
        #knn.massSimilarity(int(sys.argv[16]),int(sys.argv[17]))
        knn2 = nn.NearestNeighbor(training_set, k2)
        knn2.massSimilarity(t,phi)
        knn_missed.append(ms.testClassifier(knn, testing_set))
        knn2_missed.append(ms.testClassifier(knn2, testing_set))

        cnn = nn.NearestNeighbor(training_set, kc)
        cnn.convertToCondensed()
        #cnn.massSimilarity(int(sys.argv[18]),int(sys.argv[19]))
        cnn2 = nn.NearestNeighbor(training_set, kc2)
        cnn2.convertToCondensed()
        cnn2.massSimilarity(ct, cphi)
        cnn_missed.append(ms.testClassifier(cnn, testing_set))
        cnn2_missed.append(ms.testClassifier(cnn2, testing_set))

        enn = nn.NearestNeighbor(training_set[:validation_index], ke)
        enn.convertToEdited(training_set[validation_index:])
        #enn.massSimilarity(int(sys.argv[20]),int(sys.argv[21]))
        enn2 = nn.NearestNeighbor(training_set[:validation_index], ke2)
        enn2.convertToEdited(training_set[validation_index:])
        enn2.massSimilarity(et, ephi)
        enn_missed.append(ms.testClassifier(enn, testing_set))
        enn2_missed.append(ms.testClassifier(enn2, testing_set))
    ms.compareClassifiers(knn_missed, knn2_missed, 'knn', 'knn2')
    ms.compareClassifiers(knn_missed, cnn_missed, 'knn', 'cnn')
    ms.compareClassifiers(knn_missed, cnn2_missed, 'knn', 'cnn2')
    ms.compareClassifiers(knn_missed, enn_missed, 'knn', 'enn')
    ms.compareClassifiers(knn_missed, enn2_missed, 'knn', 'enn2')

    ms.compareClassifiers(knn2_missed, cnn_missed, 'knn2', 'cnn')
    ms.compareClassifiers(knn2_missed, cnn2_missed, 'knn2', 'cnn2')
    ms.compareClassifiers(knn2_missed, enn_missed, 'knn2', 'enn')
    ms.compareClassifiers(knn2_missed, enn2_missed, 'knn2', 'enn2')

    ms.compareClassifiers(cnn_missed, cnn2_missed, 'cnn', 'cnn2')
    ms.compareClassifiers(cnn_missed, enn_missed, 'cnn', 'enn')
    ms.compareClassifiers(cnn_missed, enn2_missed, 'cnn', 'enn2')

    ms.compareClassifiers(cnn2_missed, enn_missed, 'cnn2', 'enn')
    ms.compareClassifiers(cnn2_missed, enn2_missed, 'cnn2', 'enn2')

    ms.compareClassifiers(enn_missed, enn2_missed, 'enn', 'enn2')



if(len(sys.argv) > 15):
    chunks, class_list = openFile(sys.argv[1])
    uses_regression = False
    if sys.argv[3] == 'r':
        print("Using regression")
        uses_regression = True
    else:
        print("Using classification")

    tenFoldCV(chunks, class_list, uses_regression,
              int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7]),
              int(sys.argv[8]), int(sys.argv[9]), int(sys.argv[10]), int(sys.argv[11]),
              int(sys.argv[12]), int(sys.argv[13]), int(sys.argv[14]), int(sys.argv[15]))
    print(sys.argv)
else:
    print("Usage: <datafile> <r/c>")