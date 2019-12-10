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


a=torch.FloatTensor([[1,2],[3,4]])
b=torch.FloatTensor([[5,6],[7,8]])
print(a*b)



