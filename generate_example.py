import pandas as pd
import xml.etree.ElementTree as ET
import re
import os.path
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

example_sent = []
data = pd.read_csv('VNC-tokens', sep=" ", header=None)
df = pd.DataFrame(data)

df.columns = ['tag', 'phrase', 'routes', 'position']
df.head()
df[['L-phrase', 'R-phrase']] = df['phrase'].str.split('_', expand=True)
data = df[df.tag == 'Q']

path = 'Texts'
for filename in os.listdir(path):
    if not filename.endswith('.xml'): continue
    fullname = os.path.join(path, filename)
    tree = ET.parse(fullname)
df_nodup = data.groupby(by=['phrase']).first()

lenn = (len(data))

post_list = []
path_list = []
class_list = []
write_detail = []
classification_list = []
sentence_list = []
count = 0
for i in range(lenn):
    c = data.iloc[i, 0]
    x = data.iloc[i, 3]
    y = data.iloc[i, 2]
    post_list.append(x)
    path_list.append(y)
    class_list.append(c)


def read_txt(file_path):
    with open(file_path, "r") as f:
        return [re.split("</s>", line.rstrip('\n')) for line in f]


def join_path(path_index):
    path = 'Texts/'
    new_path = path + path_index + '.xml'
    return new_path


def get_sent(path, position, classification):
    text = read_txt(path)
    for i in range(len(text)):
        positions = 'n="' + str(position) + '"'
        if positions in str(text[i]):
            sentence_list.append(text[i])
            classification_list.append(classification)


def obtain_all(lenght):
    for i in range(lenght):
        get_sent(join_path(path_list[i]), post_list[i], class_list[i])


def parse_sent(sentence):
    sentence = str(sentence)
    soup = BeautifulSoup(sentence)
    example_sent.append(soup.text)


def get_extract():
    for i in range(len(sentence_list)):
        parse_sent(sentence_list[i])


obtain_all(lenn)
get_extract()
process_data = ""

for i in range(len(example_sent)):
    example_sent[i] = example_sent[i].replace(',', "")
    example_sent[i] = example_sent[i].replace("'", "")
    example_sent[i] = example_sent[i].replace('[', "")
    example_sent[i] = example_sent[i].replace(']', "")
    example_sent[i] = example_sent[i].replace('—', "")
    example_sent[i] = example_sent[i].replace('-', "")
    example_sent[i] = example_sent[i].replace('‘', "")
    example_sent[i] = example_sent[i].replace('’ ', "")

for i in range(len(example_sent)):
    process_data += example_sent[i] + '\n'

save_path = 'Example_sentences.txt'
if os.path.exists(save_path):
    update_file = open(save_path, 'w')
else:
    update_file = open(save_path, 'x')
    update_file = open(save_path, 'w')
update_file.write(process_data)

print(process_data)
