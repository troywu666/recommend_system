'''
@Description: 
@Version: 1.0
@Autor: Troy Wu
@Date: 2020-05-12 10:04:59
LastEditors: Troy Wu
LastEditTime: 2020-09-20 21:12:52
'''
import jieba
import math
import jieba.analyse

class TF_IDF:
    def __init__(self, file, stop_file):
        self.file = file
        self.stop_file = stop_file
        self.stop_words = self.getStopWords()

    def getStopWords(self):
        swlist = list()
        for line in open(self.stop_file, 'r', encoding = 'utf-8').readlines():
            swlist.append(line.strip())
        return swlist

    
    