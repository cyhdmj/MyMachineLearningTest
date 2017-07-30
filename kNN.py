# -*- coding: utf-8 -*-
'''
@author: wepon
@github: https://github.com/wepe
@blog:   http://blog.csdn.net/u012162613
'''
#!/usr/bin/python  E:\pythonProject\MachineLearning-master\kNN\use Python and NumPy\kNN.py
#-*-coding:utf-8-*-
from numpy import *
import operator
from os import listdir
import os

'''
#分类主体程序，计算欧式距离，选择距离最小的k个，返回k个中出现频率最高的类别
#inX是所要测试的向量  
#dataSet是训练样本集，一行对应一个样本。dataSet对应的标签向量为labels  
#k是所选的最近邻数目  
'''
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]  #shape[0]得出dataSet的行数，即样本个数                
    diffMat = tile(inX, (dataSetSize,1)) - dataSet #tile(A,(m,n))将数组A作为元素复制m行，构造出m行n列的数组  
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)    #array.sum(axis=1)按行累加，axis=0为按列累加               
    distances = sqDistances**0.5
    #sortedDistIndicies[0]表示排序后排在第一个的那个数在原来数组中的下标
    #array.argsort()，得到每个元素的排序序号
    sortedDistIndicies = distances.argsort()    
    classCount={}          #classCount是一个字典集合。
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        #get(key,x)从字典中获取key对应的value，没有key的话返回0
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    #sorted()函数，按照第二个元素即value的次序逆向（reverse=True）排序
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

'''
转化为1*1024的特征向量。程序中的filename是文件名，比如3_3.txt
样本是32*32的二值图片，将其处理成1*1024的特征向量
'''
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

'''
相当于主函数
将训练集图片合并成100*1024的大矩阵，同时逐一对测试集中的样本分类
'''
def handwritingClassTest():

    hwLabels = []
    #os模块中的listdir('str')可以读取目录str下的所有文件名，返回一个字符串列表 
    trainingFileList = listdir('trainingDigits')          
    m = len(trainingFileList)
    #加载训练集到大矩阵trainingMat  
    trainingMat = zeros((m,1024))
    for i in range(m):
        #训练样本的命名格式：1_120.txt
        fileNameStr = trainingFileList[i]  
        #string.split('str')以字符str为分隔符切片，返回list，这里去list[0],得到类似1_120这样的                 
        fileStr = fileNameStr.split('.')[0]    
        #以_切片，得到1，即类别            
        classNumStr = int(fileStr.split('_')[0])          
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    
    #逐一读取测试图片，同时将其分类 
    testFileList = listdir('testDigits')       
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 2)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
        if (classifierResult != classNumStr): errorCount += 1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(mTest))

#print os.path.abspath('.') 该函数默认打印的是Canopy的安装目录C:\Users\zhangziyang

os.chdir('E:/pythonProject/MachineLearning-master/kNN/use Python and NumPy')
handwritingClassTest()