import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from numpy import argmax
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold,cross_val_predict,cross_validate
from sklearn.svm import LinearSVC
from sklearn.metrics import log_loss
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split  # 训练测试数据分割
from sklearn.metrics import classification_report  #预测评估 
from sklearn.preprocessing import StandardScaler  # 标准化工具  
from sklearn.svm import LinearSVC
import jieba
import math
from MLE import MLE
from sklearn.feature_extraction.text import  CountVectorizer

#读取excel
def read_excel(filefullpath):    
    df = pd.read_excel(filefullpath)
    #print (df.columns.values.tolist()) #得到dataframe的keyset
    return df

#训练模型
def word2vec_word_train_value(df):
    sentences = df['就诊情况']
    #print(sentences)
    line_sent = []
    for s in sentences:
        s = s.replace("。","").replace("，",",").split(",")
        #print(s)
        line_sent.append(s)  #句子组成list
    n_dim=50
    model = Word2Vec(line_sent, 
                    size=n_dim, 
                    window=5,
                    min_count=1, 
                    workers=2,sg=1, hs=1)
    model.save('./word2vec.model')
    a = vec = np.zeros(n_dim).reshape((1, n_dim))
    for s in sentences:
        text = s.replace("。","").replace("，",",").split(",")
        vec = np.zeros(n_dim).reshape((1, n_dim))
        # print (type(vec))
        count = 0.
        for word in text:
            try:
                vec += model[word].reshape((1, n_dim))
                count += 1.
            except KeyError:
                continue
        if count != 0:
            vec /= count
        a = np.vstack((a,vec))
    x = a[1:]   #train_data
    return x

def word2vec_character_train_value(df):
    sentences = df['就诊情况']
    #print(sentences)
    line_sent = []
    for s in sentences:
        s = s.replace("。","").replace("，","").replace(",","")
        list1=[]
        for x in s:
            list1.append(x)  #句子组成list
        line_sent.append(list1) 
    #print (line_sent)
    n_dim=100
    model = Word2Vec(line_sent, 
                    size=n_dim, 
                    window=5,
                    min_count=1, 
                    workers=2,sg=1, hs=1)
    model.save('./word2vec.model')
    a = vec = np.zeros(n_dim).reshape((1, n_dim))
    for s in sentences:
        text = s.replace("。","").replace("，",",").split(",")
        vec = np.zeros(n_dim).reshape((1, n_dim))
        # print (type(vec))
        count = 0.
        for word in text:
            try:
                vec += model[word].reshape((1, n_dim))
                count += 1.
            except KeyError:
                continue
        if count != 0:
            vec /= count
        a = np.vstack((a,vec))
    x = a[1:]   #train_data
    return x

def word2vec_jieba_train_value(df):
    sentences = df['就诊情况']
    line_sent = []
    for s in sentences:
        s = s.replace("。","").replace("，","").replace(",","")
        seg_list = jieba.cut(s, cut_all=True, HMM=False)
        #print(list(seg_list))  # 全模式
        line_sent.append(list(seg_list)) 
   # print (line_sent)
    n_dim=100
    model = Word2Vec(line_sent, 
                    size=n_dim, 
                    window=5,
                    min_count=1, 
                    workers=2,sg=1, hs=1)
    model.save('./word2vec.model')
    a = vec = np.zeros(n_dim).reshape((1, n_dim))
    for s in sentences:
        text = s.replace("。","").replace("，",",").split(",")
        vec = np.zeros(n_dim).reshape((1, n_dim))
        # print (type(vec))
        count = 0.
        for word in text:
            try:
                vec += model[word].reshape((1, n_dim))
                count += 1.
            except KeyError:
                continue
        if count != 0:
            vec /= count
        a = np.vstack((a,vec))
    x = a[1:]   #train_data
    return x
#词袋模型
def vsm(df):
    sentences = df['就诊情况']
    vectorizer=CountVectorizer()
    line_list = vectorizer.fit_transform(sentences).todense()#todense将稀疏矩阵转化为完整特征矩阵
    return line_list

def onehot_train_label(df):   
#构建train_label,onehot编码，对所有的值统一编码，505种
    d = df['中医疾病']
    label_list = []
    for i in d:
        #print(i)
        label_list.append(i)

    #print (label_list)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(label_list)
    print(integer_encoded)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    #print(onehot_encoded)
    y = onehot_encoded
    #y = integer_encoded
    # inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
    # print(inverted)
    return y

#浅层模型训练
def model_train(x,y):
    X_train, X_test,y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.8)
    #x_test,y_test=loadtest("test_data.csv")
    print (X_train.shape,y_train.shape,X_test.shape,y_test.shape)
    # SVC 模型
    svc = SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr',probability=True)
    
    # 逻辑回归模型
    LR = LogisticRegression(solver='sag')

    # 朴素贝叶斯
    nb = MultinomialNB()

    # 集成模型
    # ExtraTreeClass
    etc = ExtraTreesClassifier()

    # RandomForest
    rfc = RandomForestClassifier(n_estimators=160)
    
   
    gbc = GradientBoostingClassifier(n_estimators=100,max_depth=5,min_samples_split=700)


    ada = AdaBoostClassifier()
    
    models = [svc,LR,nb,etc,rfc,gbc,ada]
    for i in models:
        model = gbc
        model.fit(X_train, y_train.ravel())
        Y_predict=model.predict(X_test)
        print(model)
        print (u"算法预测准确度为：%f%%"%np.mean((Y_predict == y_test.ravel())*100))
       # scores = cross_validate(model,X_train,y_train,scoring='neg_log_loss',cv=5)
       # print(scores)
   # print (classification_report(y_test.ravel(), Y_predict))
#计算loss
    # pred = model.predict_proba(X_test)
    # score = log_loss(y_test,pred)
    # print("loss:",score)
    # print(model)




filefullpath = r"123.xlsx"
df = read_excel(filefullpath)
x = vsm(df)
#x = word2vec_jieba_train_value(df) #jieba
#x = word2vec_character_train_value(df)#按字
# x = word2vec_word_train_value(df)#按词
y = onehot_train_label(df)
MLE(x,y)
#model_train(x,y)

