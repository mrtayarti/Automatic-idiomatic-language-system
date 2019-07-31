from gensim.models import Word2Vec
import re
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

data = pd.read_csv('Dataset.csv', encoding="latin-1")
data = data.drop([0, 0])
data.columns = ['Sentence', 'Idiom', 'NumOfWordsBetween', 'BetweenPOS', 'PrePOS', 'prePOS2', 'postPOS',
                'postPOS2', 'CountOfSameNounInContext', 'SentenceLength', 'SentimentsAVG', 'Usage']

data = data.applymap(str)
train = data[data.Usage != 'Q']
X = train.iloc[0:, 0:11].values
Y = train.iloc[0:, 11].values
X = X.tolist()
w2v_voc = data.iloc[0:, 0:11].values
w2v_voc = w2v_voc.tolist()

model = Word2Vec(w2v_voc, min_count=1)
# summarize vocabulary
words = list(model.wv.vocab)
# access vector for one word
jj = []
for i in X:
    jj.append(model[i])
# save model
model.save('model.bin')
# load model
new_model = Word2Vec.load('model.bin')
jj = np.asarray(jj)
nsamples, nx, ny = jj.shape
jj = jj.reshape((nsamples, nx * ny))

x_train, x_test, y_train, y_test = train_test_split(jj, Y, test_size=0.50, random_state=0)


parameter_grid = {"C": np.logspace(0, 4, 10), "penalty": ["l1", "l2"]}

logist = GridSearchCV(LogisticRegression(random_state=0, class_weight='balanced', max_iter = 5000), parameter_grid, cv=5)

# logist = LogisticRegression()

print("====================Grid Search================\n", logist.fit(x_train, y_train))
print("\n====================Kappa Statistic====================\n", cohen_kappa_score(y_test, logist.predict(x_test)))
print("\n====================Confusion Matrix====================\n", pd.crosstab(y_test, logist.predict(x_test),
                                                                                  rownames=['True'],
                                                                                  colnames=['Predicted'], margins=True))
print("\n====================Precision table====================\n",
      classification_report(y_test, logist.predict(x_test)))
print("\n====================Accuracy====================\n ", accuracy_score(y_test, logist.predict(x_test)))


def read_txt(file_path):
    with open(file_path, "r") as f:
        return [re.split("</s>", line.rstrip('\n')) for line in f]

print("""\n=================== Select Menu ====================
1.Load example  and get prediction
E.EXIT""")
key = '0'
key = key.replace(" ", "")  # remove space from 'key'
key = key.upper()  # turn letter in to UPPERCASE
ans = True  # set 'ans' as True
while ans:  # infinity loop to display selection menu using while loop
    ans = input("Enter choice :")
    if ans == "1":
        example = read_txt('Example_sentences.txt')
        test = data[data.Usage == 'Q']
        sentt = test.iloc[0:, 0].values
        test = test.iloc[0:, 0:11].values
        test = test.tolist()
        print('\n==============Running over', len(sentt), 'example Sentences.===============')
        test_vec = []
        for i in test:
            test_vec.append(new_model[i])
        test_vec = np.asarray(test_vec)
        nsamples, nx, ny = test_vec.shape
        test_vec = test_vec.reshape((nsamples, nx * ny))
        counter = 0
        for i in logist.predict(test_vec):
            if counter < len(logist.predict(test_vec)):
                print(i, ' : ', ', '.join(example[counter]))
            counter += 1
    elif ans == "E":
        break  # break the loop when ans = "E"



