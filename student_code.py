import math
import re

##### STILL TO ACCOUNT FOR
#       word not seen in training data

##### DATA STRUCTURES
#       vocab = dict{word: (numPositive, numTotal)}


class Bayes_Classifier:

    def __init__(self):
        self.vocab = dict()
        self.numPositiveExamples = 0
        self.numTotalExamples = 0
        self.numPosOccurrences = 0
        self.numNegOccurrences = 0
        self.uniqueWords = 0
        self.stopwords = {"ourselves", "hers", "between", "yourself", "but", "again", "there", "about", "once", "during", "out", "very",
            "having", "with", "they", "own", "an", "be", "some", "for", "do", "its", "yours", "such", "into", "of", "most", "itself", "other",
            "off", "is", "s", "am", "or", "who", "as", "from", "him", "each", "the", "themselves", "until", "below", "are", "we", "these",
            "your", "his", "through", "don", "nor", "me", "were", "her", "more", "himself", "this", "down", "should", "our", "their",
            "while", "above", "both", "up", "to", "ours", "had", "she", "all", "no", "when", "at", "any", "before", "them", "same",
            "and", "been", "have", "in", "will", "on", "does", "yourselves", "then", "that", "because", "what", "over", "why", "so",
            "can", "did", "not", "now", "under", "he", "you", "herself", "has", "just", "where", "too", "only", "myself", "which",
            "those", "i", "after", "few", "whom", "t", "being", "if", "theirs", "my", "against",
            "a", "by", "doing", "it", "how", "further", "was", "here", "than"}


    def train(self, lines):
        self.bagOfWords(lines)

    def classify(self, lines):
        self.uniqueWords = len(self.vocab)
        predictions = []
        logProbPositive = math.log(self.numPositiveExamples / self.numTotalExamples)
        logProbNegative = math.log((self.numTotalExamples - self.numPositiveExamples) / self.numTotalExamples)
        for line in lines:
            line = line.replace('\n', '')
            fields = line.split('|')
            sentiment = fields[0]
            wID = int(fields[1])
            text = fields[2]
            words = self.preprocessing(text)
            logProbPos = logProbPositive
            logProbNeg = logProbNegative
            lengthWords = len(words)
            for word in words:

                if word in self.stopwords:
                    pass
                if word in self.vocab:

                    #### fix smoothing here
                    if (self.vocab[word][0] == 0):
                        logProbPos += math.log(1 / (self.numPosOccurrences + self.uniqueWords))
                        logProbNeg += math.log((self.vocab[word][1] - self.vocab[word][0] + 1) / (
                                    self.numNegOccurrences + self.uniqueWords))
                    elif (self.vocab[word][1] - self.vocab[word][0]== 0):
                        logProbPos += math.log((self.vocab[word][0] + 1) / (self.numPosOccurrences + self.uniqueWords))
                        logProbNeg += math.log(1 / (self.numNegOccurrences + self.uniqueWords))
                    else:
                        logProbPos += math.log((self.vocab[word][0] + 1) / (self.numPosOccurrences + self.uniqueWords))
                        logProbNeg += math.log((self.vocab[word][1] - self.vocab[word][0] + 1) / (self.numNegOccurrences + self.uniqueWords))
                else:
                    logProbPos += math.log(1 / (self.numPosOccurrences + self.uniqueWords))
                    logProbNeg += math.log(1 / (self.numNegOccurrences + self.uniqueWords))

            if (logProbPos > logProbNeg):
                predictions.append('5')
            else:
                predictions.append('1')

        return predictions

    def preprocessing(self, txt):
        txt = txt.lower()
        txt = re.sub("[^a-zA-Z]", " ", txt)
        txt = re.sub(' +', ' ', txt)
        words = txt.split(' ')
        for i in range(0 , len(words)):
            words[i] = words[i][:5]
        return words



    def bagOfWords(self, lines):
        for line in lines:
            line = line.replace('\n', '')
            fields = line.split('|')
            sentiment = fields[0]
            wID = int(fields[1])
            text = fields[2]
            words = self.preprocessing(text)
            if (sentiment == '5'):
                self.numPositiveExamples += 1
            self.numTotalExamples += 1
            for word in words:
                if word not in self.vocab:
                    if (sentiment == '1'):
                        self.numNegOccurrences += 1
                        self.vocab[word] = (0, 1)
                    else:
                        self.numPosOccurrences += 1
                        self.vocab[word] = (1, 1)
                else:
                    if (sentiment == '1'):
                        self.numNegOccurrences += 1
                        self.vocab[word] = (self.vocab[word][0], self.vocab[word][1] + 1)
                    else:
                        self.numPosOccurrences += 1
                        self.vocab[word] = (self.vocab[word][0] + 1, self.vocab[word][1] + 1)
