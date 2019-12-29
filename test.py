#coding=utf-8
from math import log
import operator
import numpy as np
import pickle
f=open("myTree.txt","rb")
myTree=pickle.load(f)
if type(myTree)==dict:
    print(1)

