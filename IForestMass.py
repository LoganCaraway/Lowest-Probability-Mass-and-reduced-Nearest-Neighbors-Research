import random
import copy
import math

class IForestMass:

    def __init__(self, data, num_trees, subsample_size, num_features):
        #self.D = len(data)
        #self.r = random.seed()
        self.trees = []
        self.final_feature_index = num_features-1
        # max tree height
        h = math.ceil(math.log2(subsample_size))
        for i in range(num_trees):
            # build initial trees
            random.shuffle(data)
            r_sample = data[:subsample_size]
            self.trees.append(self.iTreeMass(r_sample, 0, h))
            # calculate total mass for nodes in trees using full data sets
            self.buildTreeMass(self.trees[i], data)

    # x: input data
    # e: current height
    # h: height limit
    # build an iTree using mass-based dissimilarity
    def iTreeMass(self, data, e, h):
        if (e >= h) or (len(data) <= 1):
            return [len(data), data, None, None, None, None]
        else:
            # randomly select attribute
            attribute_q = random.randint(0,self.final_feature_index)
            # get min and max and randomly select a split between them for this attribute
            min = data[0][attribute_q]
            max = data[0][attribute_q]
            for example in range(len(data)):
                if data[example][attribute_q] < min:
                    min = data[example][attribute_q]
                elif data[example][attribute_q] > max:
                    max = data[example][attribute_q]
            q_split = random.uniform(min, max)
            # elements less than the split go into the left set, elements greater go in the right split
            data_left = []
            data_right = []
            for example in range(len(data)):
                if data[example][attribute_q] <= q_split:
                    data_left.append(copy.deepcopy(data[example]))
                else:
                    data_right.append(copy.deepcopy(data[example]))
            left = self.iTreeMass(data_left, e+1, h)
            right = self.iTreeMass(data_right, e+1, h)
            return [len(data), data, left, right, attribute_q, q_split]

    def buildTreeMass(self, tree, data):
        tree[0] = len(data)
        tree[1] = data
        if (not tree[2] is None) and (not tree[3] is None):
            data_left = []
            data_right = []
            for example in range(len(data)):
                if data[example][tree[4]] <= tree[5]:
                    data_left.append(copy.deepcopy(data[example]))
                else:
                    data_right.append(copy.deepcopy(data[example]))
            self.buildTreeMass(tree[2], data_left)
            self.buildTreeMass(tree[3], data_right)

    # return similarity between two points within a given tree
    # the similarity is the mass of the deepest node containing both obs1 and obs2
    def massDissimilarityForTree(self, tree, obs1, obs2):
        mass = tree[0]
        if (not tree[2] is None) and (not tree[3] is None):
            if (obs1[tree[4]] <= tree[5]) and (obs2[tree[4]] <= tree[5]):
                mass = self.massDissimilarityForTree(tree[2], obs1, obs2)
            elif (obs1[tree[4]] > tree[5]) and (obs2[tree[4]] > tree[5]):
                mass = self.massDissimilarityForTree(tree[3], obs1, obs2)
        return mass

    # return similarity between two points over the whole iForest
    def getMassDissimilarity(self, obs1, obs2):
        similarity = 0
        for t in range(len(self.trees)):
            similarity += self.massDissimilarityForTree(self.trees[t], obs1, obs2)
        #similarity /= len(self.trees)
        #similarity /= self.D
        # I do not divide by number of trees or normalize over size of data set since we are looking at relative
        # differences, so these cancel
        return similarity