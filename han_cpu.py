from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
from bs4 import BeautifulSoup
import itertools
import more_itertools
import numpy as np
import pickle
from gensim.models import Word2Vec
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import string
import warnings
import time
import math
warnings.filterwarnings('ignore')


df=pd.read_csv('yelp.csv')
#print(df.head())

col_text='text'
col_target='cool'

#下面是对评分排序去重，然后得出不同等级的个数
cls_arr=np.sort(df[col_target].unique()).tolist() #普通的list类型
#print(cls_arr)
#print(type(cls_arr))
classes=len(cls_arr)


length=df.shape[0]
train_len=int(0.8*length)
val_len=int(0.1*length)

train=df[:train_len]
val=df[train_len:train_len+val_len]
test=df[train_len+val_len:]

def clean_str(string, max_seq_len):
    string = BeautifulSoup(string, "lxml").text
    string = re.sub(r"[^A-Za-z0-9(),!?\"\`]", " ", string)
    string = re.sub(r"\"s", " \"s", string)
    string = re.sub(r"\"ve", " \"ve", string)
    string = re.sub(r"n\"t", " n\"t", string)
    string = re.sub(r"\"re", " \"re", string)
    string = re.sub(r"\"d", " \"d", string)
    string = re.sub(r"\"ll", " \"ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    s = string.strip().lower().split(" ")
    if len(s) > max_seq_len:
        return s[0:max_seq_len]
    return s

def create3DList(df,col,max_sent_len,max_seq_len):
    x=[]
    for docs in df[col].as_matrix():
        x1=[]
        idx=0
        for seq in "|||".join(re.split("[.?!]",docs)).split("|||"):
            x1.append(clean_str(seq,max_sent_len))
            if idx>=max_seq_len-1:
                break
            idx=idx+1
        x.append(x1)
    return x

max_sent_len=2
max_seq_len=3

x_train=create3DList(train,col_text,max_sent_len,max_seq_len)
x_val=create3DList(val,col_text,max_sent_len,max_seq_len)
x_test=create3DList(test,col_text,max_sent_len,max_seq_len)

#print("x_train's type is")
#print(type(x_train))



stoplist = stopwords.words('english') + list(string.punctuation)
stemmer = SnowballStemmer('english')
x_train_texts = [[[stemmer.stem(word.lower()) for word in sent if word not in stoplist] for sent in para]
                 for para in x_train]
x_test_texts = [[[stemmer.stem(word.lower()) for word in sent if word not in stoplist] for sent in para]
                for para in x_test]
x_val_texts = [[[stemmer.stem(word.lower()) for word in sent if word not in stoplist] for sent in para]
               for para in x_val]

## calculate frequency of words
#移除频率少于5的词语
from collections import defaultdict  #写重复了，和下面一段，待会去掉

frequency1 = defaultdict(int)
for texts in x_train_texts:
    for text in texts:
        for token in text:
            frequency1[token] += 1
for texts in x_test_texts:
    for text in texts:
        for token in text:
            frequency1[token] += 1
for texts in x_val_texts:
    for text in texts:
        for token in text:
            frequency1[token] += 1

## remove  words with frequency less than 5.
#下面的列表是三维的，第一维度表示多篇文章，第二维表示文章里的多个句子，第三维表示一句话里面的所有词
x_train_texts = [[[token for token in text if frequency1[token] > 5]
                  for text in texts] for texts in x_train_texts]
x_test_texts = [[[token for token in text if frequency1[token] > 5]
                 for text in texts] for texts in x_test_texts]
x_val_texts = [[[token for token in text if frequency1[token] > 5]
                for text in texts] for texts in x_val_texts]

texts=list(more_itertools.collapse(x_train_texts[:]+x_test_texts[:]+x_val_texts[:],levels=1))#把列表里的所有元素都列出来变成一个扁平列表，levels表示去掉几层[]


word2vec=Word2Vec(texts,size=200,min_count=5)
word2vec.save("dictonary_yelp")

## convert 3D text list to 3D list of index
#这里每个单词存的不是词向量，而是词语所在的训练好的word2vec词向量字典里面的编号
x_train_vec = [[[word2vec.wv.vocab[token].index for token in text]
         for text in texts] for texts in x_train_texts]

x_test_vec = [[[word2vec.wv.vocab[token].index for token in text]
         for text in texts] for texts in x_test_texts]

x_val_vec = [[[word2vec.wv.vocab[token].index for token in text]
         for text in texts] for texts in x_val_texts]

weights=torch.FloatTensor(word2vec.wv.syn0) #存的是所有的词向量[[词向量],[],[]]
#print(weights.size()) (4681,200)
vocab_size=len(word2vec.wv.vocab)
#print(vocab_size)  4681
y_train = train[col_target].tolist()
y_test = test[col_target].tolist()
y_val = val[col_target].tolist()


def attention_mul(rnn_outputs, att_weights):
    attn_vectors = None
    for i in range(rnn_outputs.size(0)):
        h_i = rnn_outputs[i]
        a_i = att_weights[i]
        h_i = a_i * h_i #表示对应位置数相乘，不是矩阵乘法
        h_i = h_i.unsqueeze(0)
        if(attn_vectors is None):
            attn_vectors = h_i
        else:
            attn_vectors = torch.cat((attn_vectors,h_i),0)
    return torch.sum(attn_vectors, 0).unsqueeze(0)

## The word RNN model for generating a sentence vector
class WordRNN(nn.Module):
    def __init__(self, vocab_size,embedsize, batch_size, hid_size):
        super(WordRNN, self).__init__()
        self.batch_size = batch_size
        self.embedsize = embedsize #200
        self.hid_size = hid_size
        ## Word Encoder
        self.embed = nn.Embedding(vocab_size, embedsize)
        self.wordRNN = nn.GRU(embedsize, hid_size, bidirectional=True)
        ## Word Attention
        self.wordattn = nn.Linear(2*hid_size, 2*hid_size)
        self.attn_combine = nn.Linear(2*hid_size, 2*hid_size,bias=False)
    def forward(self,inp, hid_state):
        emb_out  = self.embed(inp)

        out_state, hid_state = self.wordRNN(emb_out, hid_state)
        # out_state.size() (12,64,200)  seq_len,batch,num_directions*hidden_size
        # hid_state.size() (2,64,100)  num_layers*num_directions,batch,hidden_size

        word_annotation = self.wordattn(out_state) # (12,64,200)
        attn = F.softmax(self.attn_combine(word_annotation),dim=1)
        #print(attn)

        sent = attention_mul(out_state,attn)
        return sent, hid_state

class SentenceRNN(nn.Module):
    def __init__(self, vocab_size, embedsize, batch_size, hid_size, c):
        super(SentenceRNN, self).__init__()
        self.batch_size = batch_size
        self.embedsize = embedsize
        self.hid_size = hid_size
        self.cls = c
        self.wordRNN = WordRNN(vocab_size, embedsize, batch_size, hid_size)
        ## Sentence Encoder
        self.sentRNN = nn.GRU(embedsize, hid_size, bidirectional=True)
        ## Sentence Attention
        self.sentattn = nn.Linear(2 * hid_size, 2 * hid_size)
        self.attn_combine = nn.Linear(2 * hid_size, 2 * hid_size, bias=False)
        self.doc_linear = nn.Linear(2 * hid_size, c)

    def forward(self, inp, hid_state_sent, hid_state_word):
        s = None
        ## Generating sentence vector through WordRNN
        for i in range(len(inp[0])):
            r = None
            for j in range(len(inp)):
                if (r is None):
                    r = [inp[j][i]]
                else:
                    r.append(inp[j][i])
            r1 = np.asarray([sub_list + [0] * (max_seq_len - len(sub_list)) for sub_list in r]) #r1.shape()  (64,12)
            _s, state_word = self.wordRNN(torch.LongTensor(r1).view(-1, batch_size), hid_state_word)#batch_size 64
            #print(_s.size()) (1,64,200)
            #print(state_word.size())#(2,64,200) 注意，这里的200是num_directions*hidden_size 即2*100,可以推出hidden_size就是为了凑句子向量也为200维而设置的
            if (s is None):
                s = _s
            else:
                s = torch.cat((s, _s), 0)
        out_state, hid_state = self.sentRNN(s, hid_state_sent)
        #print(s.size()) (25,64,200)
        #print(out_state.size()) (25,64,200)
        #print(hid_state.size()) (2,64,100)
        sent_annotation = self.sentattn(out_state)
        #print(self.attn_combine(sent_annotation))
        attn = F.softmax(self.attn_combine(sent_annotation), dim=1)

        doc = attention_mul(out_state, attn)
        d = self.doc_linear(doc)
        cls = F.log_softmax(d.view(-1, self.cls), dim=1)
        #print(cls.size())  #(64,29)
        return cls, hid_state

    def init_hidden_sent(self):
        return Variable(torch.zeros(2, self.batch_size, self.hid_size))

    def init_hidden_word(self):
        return Variable(torch.zeros(2, self.batch_size, self.hid_size))



y_train_tensor=[torch.FloatTensor([cls_arr.index(label)]) for label in y_train ]
y_val_tensor=[torch.FloatTensor([cls_arr.index(label)]) for label in y_val]
y_test_tensor=[torch.FloatTensor([cls_arr.index(label)]) for label in y_test]

max_seq_len=max([len(seq) for seq in itertools.chain.from_iterable(x_train_vec+x_val_vec+x_test_vec)]) #类似more_itertools.collapse,但是是迭代去除一层[]的所有元素,这里选取的是所有的句子
max_sent_len=max([len(sent) for sent in (x_train_vec +x_val_vec+x_test_vec)])

# print(max_seq_len) 12 表示一句话中包含词语的最大数量
# print(max_sent_len)  25 表示一个文档中包含句子的最大数量
# print(type(max_seq_len))  <class 'int'>
# print(np.percentile(np.array([len(seq) for seq in itertools.chain.from_iterable(x_train_vec +x_val_vec + x_test_vec)]),90))   6.0
# print(np.percentile(np.array([len(sent) for sent in (x_train_vec +x_val_vec + x_test_vec)]),90))   25.0


#下面的X_.._pad是补全0使得每一篇文档里的句子数量都是最大的文档对应的句子数量,维度还是三维的[[[' ',' ' , , ,]],[[]]]
X_train_pad = [sub_list + [[0]] * (max_sent_len - len(sub_list)) for sub_list in x_train_vec]
X_val_pad = [sub_list + [[0]] * (max_sent_len - len(sub_list)) for sub_list in x_val_vec]
X_test_pad = [sub_list + [[0]] * (max_sent_len - len(sub_list)) for sub_list in x_test_vec]



batch_size=64

def train_data(batch_size, review, targets, sent_attn_model, sent_optimizer, criterion):
    state_word = sent_attn_model.init_hidden_word()
    state_sent = sent_attn_model.init_hidden_sent()
    sent_optimizer.zero_grad()

    y_pred, state_sent = sent_attn_model(review, state_sent, state_word)

    loss = criterion(y_pred, torch.LongTensor(targets))

    max_index = y_pred.max(dim=1)[1]
    correct = (max_index == torch.LongTensor(targets)).sum()
    acc = float(correct) / batch_size

    loss.backward()

    sent_optimizer.step()

    #return loss.data[0], acc
    return loss.data, acc

hid_size = 100
embedsize = 200


sent_attn = SentenceRNN(vocab_size,embedsize,batch_size,hid_size,classes) #(4681,200,64,100,29)
#sent_attn.cuda()
sent_attn.wordRNN.embed.from_pretrained(weights)#传递预处理好的word2vec词向量权重
#torch.backends.cudnn.benchmark=True



learning_rate = 1e-3
momentum = 0.9
sent_optimizer = torch.optim.SGD(sent_attn.parameters(), lr=learning_rate, momentum= momentum)
criterion = nn.NLLLoss()

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def gen_batch(x,y,batch_size):
    k = random.sample(range(len(x)-1),batch_size) #random.sample(sequence,k) 从指定序列中随机获取k个元素作为一个片段返回，sample函数不会修改原有序列
    x_batch=[]
    y_batch=[]

    for t in k:
        x_batch.append(x[t])
        y_batch.append(y[t])
    return [x_batch,y_batch]


def validation_accuracy(batch_size, x_val, y_val, sent_attn_model):
    acc = []
    val_length = len(x_val)
    for j in range(int(val_length / batch_size)):
        x, y = gen_batch(x_val, y_val, batch_size)
        state_word = sent_attn_model.init_hidden_word()
        state_sent = sent_attn_model.init_hidden_sent()

        y_pred, state_sent = sent_attn_model(x, state_sent, state_word)
        max_index = y_pred.max(dim=1)[1]
        correct = (max_index == torch.LongTensor(y)).sum()
        acc.append(float(correct) / batch_size)
    return np.mean(acc)

def test_accuracy(batch_size, x_test, y_test, sent_attn_model):
    acc = []
    test_length = len(x_test)
    for j in range(int(test_length / batch_size)):
        x, y = gen_batch(x_test, y_test, batch_size)
        state_word = sent_attn_model.init_hidden_word()
        state_sent = sent_attn_model.init_hidden_sent()

        y_pred, state_sent = sent_attn_model(x, state_sent, state_word)
        max_index = y_pred.max(dim=1)[1]
        correct = (max_index == torch.LongTensor(y)).sum()
        acc.append(float(correct) / batch_size)
    return np.mean(acc)

def train_early_stopping(batch_size, x_train, y_train, x_val, y_val, sent_attn_model,
                         sent_attn_optimiser, loss_criterion, num_epoch,
                         print_loss_every=50, code_test=True):
    start = time.time()
    loss_full = []
    loss_epoch = []
    acc_epoch = []
    acc_full = []
    val_acc = []
    epoch_counter = 0
    train_length = len(x_train)
    for i in range(1, num_epoch + 1): #这里每一个epoch都是独立的
        loss_epoch = []
        acc_epoch = []
        for j in range(int(train_length / batch_size)):
            x, y = gen_batch(x_train, y_train, batch_size)
            loss, acc = train_data(batch_size, x, y, sent_attn_model, sent_attn_optimiser, loss_criterion)
            loss_epoch.append(loss)
            acc_epoch.append(acc)
            if (code_test and j % int(print_loss_every / batch_size) == 0):
                print('Loss at %d paragraphs, %d epoch,(%s) is %f' % (
                j * batch_size, i, timeSince(start), np.mean(loss_epoch)))
                print('Accuracy at %d paragraphs, %d epoch,(%s) is %f' % (
                j * batch_size, i, timeSince(start), np.mean(acc_epoch)))

        loss_full.append(np.mean(loss_epoch))
        acc_full.append(np.mean(acc_epoch))
        torch.save(sent_attn_model.state_dict(), 'sent_attn_model_yelp.pth')
        print('Loss after %d epoch,(%s) is %f' % (i, timeSince(start), np.mean(loss_epoch)))
        print('Train Accuracy after %d epoch,(%s) is %f' % (i, timeSince(start), np.mean(acc_epoch)))

        val_acc.append(validation_accuracy(batch_size, x_val, y_val, sent_attn_model))
        print('Validation Accuracy after %d epoch,(%s) is %f' % (i, timeSince(start), val_acc[-1]))
        print("This is %d epoch:"% i)
        print("the test_accuracy is=%f" % test_accuracy(batch_size, X_test_pad, y_test_tensor, sent_attn))


    return loss_full, acc_full, val_acc



epoch = 200

loss_full, acc_full, val_acc = train_early_stopping(batch_size, X_train_pad, y_train_tensor, X_val_pad,
                                                    y_val_tensor, sent_attn, sent_optimizer, criterion, epoch, 10000, False)



#print(test_accuracy(batch_size, X_test_pad, y_test_tensor, sent_attn))
