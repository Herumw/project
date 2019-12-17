import tensorflow as tf
from datetime import datetime
from data_reader import DataReader
from model import Model
from utils import read_vocab, count_parameters, load_glove
import pandas as pd
import nltk
import itertools
import pickle

a=nltk.FreqDist(["aaaaaa","bbb","ccc"])
for i in a.items():
    print(i)
for i in a:
    print(i)