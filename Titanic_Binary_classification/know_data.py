#-*-coding:utf-8-*-
import pandas as pd
import numpy as np
from pandas import Series,DataFrame
import matplotlib.pyplot as plt

data_train = pd.read_csv("train.csv")  # 得到一些信息
print(data_train.info())# 得到一些信息
print (data_train.describe()) #得到数值型数据的一些分布

fig = plt.figure()
fig.set(alpha = 0.2) #设置图表颜色alpha参数

plt.subplot2grid((2,3),(0,0))
data_train.Survived.value_counts().plot(kind='bar') #柱状图
plt.title(u"获救情况(1为获救)") #标题
plt.ylabel(u"人数")


plt.subplot2grid((2,3),(0,1))
data_train.Pclass.value_counts().plot(kind="bar")
plt.ylabel(u"人数")
plt.title(u"乘客等级分布")

plt.subplot2grid((2,3),(0,2))
plt.scatter(data_train.Survived,data_train.Age)
plt.ylabel(u"年龄") #设定纵坐标名称
plt.grid(b=True,which='major',axis='y')
plt.title(u"按年龄看获救分布（1为获救）")
#
#
#
plt.subplot2grid((2,3),(1,0),colspan=2)
data_train.Age[data_train.Pclass == 1].plot(kind = 'kde')
data_train.Age[data_train.Pclass == 2].plot(kind = 'kde')
data_train.Age[data_train.Pclass == 3].plot(kind = 'kde')
plt.xlabel(u"年龄")
plt.ylabel(u"密度")
plt.title(u"各等级的乘客年龄分布")
plt.legend((u'头等舱',u'2等舱',u'3等舱'),loc='best')

plt.subplot2grid((2,3),(1,2))
data_train.Embarked.value_counts().plot(kind = 'bar')
plt.title(u"各登船口岸上船人数")
plt.ylabel(u"人数")
plt.show()

#看看各乘客等级的获救情况
fig = plt.figure()
fig.set(alpha = 0.2)
Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
df=pd.DataFrame({'获救':Survived_1,'未获救':Survived_0})
df.plot(kind = 'bar',stacked = True )

plt.title("各乘客等级的获救情况")
plt.xlabel("乘客等级")
plt.ylabel("人数")
plt.show()

#看看性别的获救情况
fig = plt.figure()
fig.set(alpha = 0.2)
Survived_f = data_train.Survived[data_train.Sex == 'male'].value_counts()
Survived_m = data_train.Survived[data_train.Sex == 'female'].value_counts()

df = pd.DataFrame({'男性':Survived_f,'女性':Survived_m})
df.plot(kind = 'bar',stacked = True)

plt.title("乘客性别的获救情况")
plt.xlabel("乘客性别")
plt.ylabel("人数")
plt.show()
##############################################################################详细情况

fig = plt.figure()
fig.set(alpha = 0.65)
plt.title("根据舱等级和性别的获救情况")
ax1 = fig.add_subplot(141)
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass !=3].value_counts().plot(kind = 'bar',label="female high class",color='#FA2479')
ax1.set_xticklabels(["获救","未获救"],rotation = 0)
ax1.legend(["女性/高级舱"],loc='best')

ax2=fig.add_subplot(142, sharey=ax1)
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='female, low class', color='pink')
ax2.set_xticklabels([u"未获救", u"获救"], rotation=0)
plt.legend([u"女性/低级舱"], loc='best')

ax3=fig.add_subplot(143, sharey=ax1)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass != 3].value_counts().plot(kind='bar', label='male, high class',color='lightblue')
ax3.set_xticklabels([u"未获救", u"获救"], rotation=0)
plt.legend([u"男性/高级舱"], loc='best')

ax4=fig.add_subplot(144, sharey=ax1)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='male low class', color='steelblue')
ax4.set_xticklabels([u"未获救", u"获救"], rotation=0)
plt.legend([u"男性/低级舱"], loc='best')

plt.show()

#################各登船港口的获救情况
fig=plt.figure()
fig.set(alpha=0.2)
Survived_0 = data_train.Embarked[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Embarked[data_train.Survived == 1].value_counts()
df=pd.DataFrame({'获救':Survived_1,'未获救':Survived_0})
df.plot(kind='bar',stacked = True)
plt.title("各登录港口乘客的获救情况")
plt.xlabel("登录港口")
plt.ylabel("人数")
plt.show()

#######################堂兄弟妹，父母/孩子有几人，对获救的影响
g = data_train.groupby(['SibSp','Survived'])
df = pd.DataFrame(g.count()['PassengerId'])
print (df)

g=data_train.groupby(['Parch','Survived'])
df = pd.DataFrame(g.count()['PassengerId'])
print (df)

##############有无Cabin信息对获救的影响
fig = plt.figure()
fig.set(alpha=0.2)
Survived_cabin = data_train.Survived[pd.notnull(data_train.Cabin)].value_counts()
Survived_nocabin = data_train.Survived[pd.isnull(data_train.Cabin)].value_counts()
df = pd.DataFrame({'有':Survived_cabin,'没有':Survived_nocabin})
df.plot(kind = 'bar',stacked = True)
plt.title("有无cabin信息的获救情况")
plt.xlabel('有无Cabin信息')
plt.ylabel("人数")
plt.show()


