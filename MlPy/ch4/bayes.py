#coding:utf-8
from numpy import *
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    ##分别表示标签 #1代表侮辱性  0代表正常
    return postingList,classVec ##返回输入数据和标签向量

def createVocabList(dataSet): #将词条库转化为不重复列表
    vocabSet=set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document) #求2个集合中的并集
    return list(vocabSet)

#将词条列表转为矢量
def bagOfWords2Vec(vocabList,inputSet): #vocalList就是不重复的词条列表，inputSet为样本的不重复词条列表
    returnVec = [0]*len(vocabList) #创建矢量，长度为词条库里词条个数N，里面有N个0
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]+=1 #若样本的词条在词条库中出现，就会在矢量中对应的下标元素变为1，                                         # 最后会变成像 [0,1,0,...,0,1,1]这样的矢量
        else: print ("the word: %s is not in my Vocabulary!" %word)
    #print returnVec
    return returnVec

# 构建贝叶斯分类器，trainMatrix为多个训练样本构成的0,1矩阵 矩阵的长度为不重复词条的长度
# trainCatergory为每个样本的分类结果，就是classVec = [0,1,0,1,0,1]
def trainNBO(trainMatrix,trainCategory):
    numTrainDocs=len(trainMatrix)#统计训练集的个数 :6
    numWords=len(trainMatrix[0])#统计词条库中的词条数 :32
    pAbusive=sum(trainCategory)/float(numTrainDocs)#计算pAbusive负面情绪(用1表示的）在训练集中的频率，:1/2
    # 在这次中为1/2，防止多个概率的成绩当中的一个为0，将基数由0改为1
    p0Num=ones(numWords)#构建一个长度为numWords的array数组
    p1Num=ones(numWords)
    p0Denom=2.0
    p1Denom=2.0
    for i in range(numTrainDocs):
        if trainCategory[i] ==1:
            p1Num +=trainMatrix[i] #统计负面类别的词条数目
            p1Denom+=sum(trainMatrix[i]) #统计负面类别的词条个数，允许统计重复词条
        else:
            p0Num+=trainMatrix[i]# 统计正面类别的词条词频
            p0Denom+=sum(trainMatrix[i]) #统计负面类别的词条个数,允许统计重复词条
    p1Vect = log(p1Num/p1Denom) #计算负面类别中每一个词条的频率
    p0Vect = log(p0Num/p0Denom) #计算正面类别中每一个词条的频率
    return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    ##vec2Classify为一个样本词条矢量，形如[0,0,1,0,...,1,0]
    ##p0Vec为正面情绪的词条矢量，已经经过对数处理
    ##p0Vec为负面情绪的词条矢量，已经经过对数处理
    ##pClass1为负面情绪的频率
    p1=sum(vec2Classify * p1Vec)+log(pClass1) #根据贝叶斯公式转化，计算贝叶斯分类概率的等价值
    p0=sum(vec2Classify * p0Vec)+log(1.0-pClass1)
    if p1>p0 :
        return 1
    else:
        return 0

def testingNB():
    listOPosts , listClasses =loadDataSet() #读取训练集的语料库以及负面情绪的分布表
    myVocabList=createVocabList(listOPosts) #得到不重复的词条库
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(bagOfWords2Vec(myVocabList,postinDoc))
    p0v,p1v,pAb = trainNBO(array(trainMat),array(listClasses))
    testEntry=['love','my','dalmation']
    thisDoc=array(bagOfWords2Vec(myVocabList,testEntry))
    print (thisDoc)
    print (testEntry,'classified as:',classifyNB(thisDoc,p0v,p1v,pAb))
    testEntry=['stupid','garbage']
    thisDoc = array(bagOfWords2Vec(myVocabList,testEntry))
    print (testEntry,'classified as:',classifyNB(thisDoc,p0v,p1v,pAb))

def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*',bigString) #用正则表达式切分 \W:非单词字符 *：匹配前一个字符0或者无限次
    return [tok.lower() for tok in listOfTokens if len(tok)>2]

#垃圾自动分类器
def spamTest():
    docList=[]
    classList=[]
    fullText=[]
    for i in range(1,26): #从1到26 不包含26
        wordList=textParse(open('email/spam/%d.txt' %i).read())
        #extend 接受一个参数，这个参数总是一个 list，并且把这个 list 中的每个元素添加到原 list 中。
        #append 接受一个参数，这个参数可以是任何数据类型，并且简单地追加到 list 的尾部。
        docList.append(wordList)  #每个spam中的每一封邮件作为一个子列表添加到docList
        fullText.extend(wordList) #每个spam中的单词添加到docList 一共有一个列表
        classList.append(1)
        wordList=textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList=createVocabList(docList) ##创建包含所有不重复的词的vocabList
    trainingSet = range(50) #一个整数列表，值从0-49
    testSet=[] #测试集
    for i in range(10): #从训练集中删除已经选择的项目
        #uniform() 方法将随机生成下一个实数，它在[x,y]范围内。
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex]) #从训练集中删除已经选择的项目
    trainMat=[]
    trainClasses=[]
    for docIndex in trainingSet: #训练分类器
        #trainMat一共有40个样本，每个样本里有692个单词对应的0或1
        trainMat.append(bagOfWords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam=trainNBO(array(trainMat),array(trainClasses))
    errorCount =0
    for docIndex in testSet: #测试错误率
        wordVector = bagOfWords2Vec(vocabList,docList[docIndex])
        # classifyNB产生的类标签和真实的类标签不同
        if classifyNB(array(wordVector),p0V,p1V,pSpam)!=classList[docIndex]:
            errorCount+=1
    print ('the error rate is:',float(errorCount)/len(testSet))

def calcMostFreq(vocabList,fullText):
    import operator
    freqDict={}
    for token in vocabList: #遍历词汇表
        freqDict[token]=fullText.count(token) #统计token出现的次数，构成词典
    sortedFreq=sorted(freqDict.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedFreq[:30]
#这个跟spamTest()基本上一样，不同在于这边访问的是RSS源，
# 最后返回词汇表，以及不同分类每个词出现的概率
def stopWords():
    import re
    wordList =  open('stopWords.txt').read() # see http://www.ranks.nl/stopwords
    listOfTokens = re.split(r'\W*', wordList)
    return [tok.lower() for tok in listOfTokens]
    print ('read stop word from \'stopWords.txt\':',listOfTokens)
    return listOfTokens

def localWords(feed1,feed0): #使用两个RSS源作为参数
    import feedparser
    docList=[]
    classList=[]
    fullText=[]
    minLen = min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):
        wordList=textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList=createVocabList(docList) #创建词汇表
    stopWordsList = stopWords()
    for stopWord in stopWordsList:
        if stopWord in vocabList:
            vocabList.remove(stopWord)
    trainingSet=range(2*minLen) #创建测试集
    testSet=[]
    a=23.3
    for i in range(5):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat=[]
    trainClasses=[]
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNBO(array(trainMat),array(trainClasses))
    errorCount=0
    for docIndex in testSet:        #classify the remaining items
        wordVector = bagOfWords2Vec(vocabList, docList[docIndex])
        classifiedClass = classifyNB(array(wordVector),p0V,p1V,pSpam)
        originalClass = classList[docIndex]
        result =  classifiedClass != originalClass
        if result:
            errorCount += 1
    print ('the error rate is:',float(errorCount)/len(testSet))
    return vocabList,p0V,p1V








