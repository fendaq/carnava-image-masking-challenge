#!/usr/bin/env python
#coding:utf-8

import os

def ListFilesToTxt(dir, file, wildcard, recursion):
    exts = wildcard.split(" ")
    for root, subdirs, files in os.walk(dir):
        for name in files:
            for ext in exts:
                if(name.endswith(ext)):
                    # name = os.path.join('test128x128', name) 
                    name = name.split('.')[0]
                    file.write(name + "\n")
                    break
            if(not recursion):
                    break

def Test():
    # dir= os.getcwd() 
    dir = "/home/lhc/Projects/Kaggle-seg/data/train"
    outfile="train_aug.txt"
    wildcard = ".jpg"
    file = open(outfile,"w")
    if not file:
        print ("cannot open the file %s for writing" % outfile)
    # ListFilesToTxt(dir+'/test128x128', file,wildcard, 1)
    ListFilesToTxt(dir, file,wildcard, 1)
 
    file.close()

Test()