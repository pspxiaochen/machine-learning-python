# remove warnings
import warnings
warnings.filterwarnings('ignore')
# ---
import pandas as pd
pd.options.display.max_columns = 100
from matplotlib import pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import numpy as np
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# def set_miss_ages(df): ##暂时的方法 ，感觉有更好的。后面考虑
#     df.Age.fillna(0,inplace=True)
#     Female_size = 0
#     Male_size = 0
#     Female_age = 0
#     Male_age = 0
#     for i in range(len(df.Age)):
#         if(df.Sex[i] == 'male'):
#             Male_age += df.Age[i]
#             Male_size += 1
#         else:
#             Female_age += df.Age[i]
#             Female_size += 1
#     Female_age_mean = (int)(Female_age / Female_size)
#     Male_age_mean = (int)(Male_age / Male_size)
#     df.loc[(df.Age == 0) & (df.Sex =='female'),'Age'] = Female_age_mean
#     df.loc[(df.Age == 0) & (df.Sex == 'male'),'Age'] = Male_age_mean
#     return df
#
# train = set_miss_ages(train)
# test = set_miss_ages(test)

def get_combined_data(train,test):

    train.drop('Survived',axis=1,inplace=True)
    combined_data = train.append(test)
    combined_data.reset_index(drop=True,inplace=True)
    return combined_data
combined_data = get_combined_data(train,test)

def get_titles(df):
    df['Title'] = df.Name.map(lambda name:name.split(',')[1].split('.')[0].strip())
    Title_Dictionary = {
                        "Capt":       "Officer",
                        "Col":        "Officer",
                        "Major":      "Officer",
                        "Jonkheer":   "Royalty",
                        "Don":        "Royalty",
                        "Sir" :       "Royalty",
                        "Dr":         "Officer",
                        "Rev":        "Officer",
                        "the Countess":"Royalty",
                        "Dona":       "Royalty",
                        "Mme":        "Mrs",
                        "Mlle":       "Miss",
                        "Ms":         "Mrs",
                        "Mr" :        "Mr",
                        "Mrs" :       "Mrs",
                        "Miss" :      "Miss",
                        "Master" :    "Master",
                        "Lady" :      "Royalty"
                        }
    df['Title'] = df.Title.map(Title_Dictionary)
    return df

combined_data=get_titles(combined_data)




grouped_train = combined_data.head(891).groupby(['Sex','Pclass','Title'])
grouped_median_train = grouped_train.median()

grouped_test = combined_data.iloc[891:].groupby(['Sex','Pclass','Title'])
grouped_median_test = grouped_test.median()
def process_age(df):

    # a function that fills the missing values of the Age variable

    def fillAges(row, grouped_median):
        if row['Sex'] == 'female' and row['Pclass'] == 1:
            if row['Title'] == 'Miss':
                return grouped_median.loc['female', 1, 'Miss']['Age']
            elif row['Title'] == 'Mrs':
                return grouped_median.loc['female', 1, 'Mrs']['Age']
            elif row['Title'] == 'Officer':
                return grouped_median.loc['female', 1, 'Officer']['Age']
            elif row['Title'] == 'Royalty':
                return grouped_median.loc['female', 1, 'Royalty']['Age']

        elif row['Sex'] == 'female' and row['Pclass'] == 2:
            if row['Title'] == 'Miss':
                return grouped_median.loc['female', 2, 'Miss']['Age']
            elif row['Title'] == 'Mrs':
                return grouped_median.loc['female', 2, 'Mrs']['Age']

        elif row['Sex'] == 'female' and row['Pclass'] == 3:
            if row['Title'] == 'Miss':
                return grouped_median.loc['female', 3, 'Miss']['Age']
            elif row['Title'] == 'Mrs':
                return grouped_median.loc['female', 3, 'Mrs']['Age']

        elif row['Sex'] == 'male' and row['Pclass'] == 1:
            if row['Title'] == 'Master':
                return grouped_median.loc['male', 1, 'Master']['Age']
            elif row['Title'] == 'Mr':
                return grouped_median.loc['male', 1, 'Mr']['Age']
            elif row['Title'] == 'Officer':
                return grouped_median.loc['male', 1, 'Officer']['Age']
            elif row['Title'] == 'Royalty':
                return grouped_median.loc['male', 1, 'Royalty']['Age']

        elif row['Sex'] == 'male' and row['Pclass'] == 2:
            if row['Title'] == 'Master':
                return grouped_median.loc['male', 2, 'Master']['Age']
            elif row['Title'] == 'Mr':
                return grouped_median.loc['male', 2, 'Mr']['Age']
            elif row['Title'] == 'Officer':
                return grouped_median.loc['male', 2, 'Officer']['Age']

        elif row['Sex'] == 'male' and row['Pclass'] == 3:
            if row['Title'] == 'Master':
                return grouped_median.loc['male', 3, 'Master']['Age']
            elif row['Title'] == 'Mr':
                return grouped_median.loc['male', 3, 'Mr']['Age']

    df.head(891).Age = df.head(891).apply(lambda r: fillAges(r, grouped_median_train) if np.isnan(r['Age'])
    else r['Age'], axis=1)

    df.iloc[891:].Age = df.iloc[891:].apply(lambda r: fillAges(r, grouped_median_test) if np.isnan(r['Age'])
    else r['Age'], axis=1)
    return df

combined_data = process_age(combined_data)

def process_names(df):
    df.drop('Name',axis=1,inplace=True)
    dummies_title = pd.get_dummies(df.Title,prefix='Title')
    df = pd.concat([df,dummies_title],axis=1)
    df.drop('Title',axis = 1,inplace = True)
    return df
combined_data = process_names(combined_data)

def set_miss_fare(df):
    df.head(891).Fare.fillna(df.head(891).Fare.mean(),inplace=True)
    df.iloc[891:].Fare.fillna(df.loc[891:].Fare.mean(),inplace=True)
    df.Fare = (df.Fare - np.min(df.Fare)) / (np.max(df.Fare) - np.min(df.Fare))
    return df
combined_data=set_miss_fare(combined_data)

def set_miss_embarked(df) :
    df.head(891).Embarked.fillna('S',inplace=True)
    df.iloc[891:].Embarked.fillna('S',inplace=True)
    dummies_embarked = pd.get_dummies(df.Embarked,prefix='Embarked')
    df = pd.concat([df,dummies_embarked],axis = 1)
    df.drop('Embarked',axis = 1,inplace = True)
    return df

combined_data = set_miss_embarked(combined_data)

def set_miss_cabin(df):
    df.Cabin.fillna('U',inplace=True)
    df.Cabin = df.Cabin.map(lambda c : c[0])
    dummies_cabin = pd.get_dummies(df.Cabin,prefix='Cabin')
    df = pd.concat([df,dummies_cabin],axis=1)
    df.drop('Cabin',axis = 1,inplace = True)
    return df

combined_data = set_miss_cabin(combined_data)


def process_sex(df):
    df.Sex = df.Sex.map({'female':0,'male':1})
    return df

combined_data = process_sex(combined_data)

def process_pclass(df):
    dummies_pclass = pd.get_dummies(df.Pclass,prefix='Pclass')
    df = pd.concat([df,dummies_pclass],axis=1)
    df.drop('Pclass',axis = 1,inplace = True)
    return df

combined_data = process_pclass(combined_data)

def process_ticket(df):
    def cleanTicket(ticket):
        ticket = ticket.replace('.','')
        ticket = ticket.replace('/','')
        ticket = ticket.split()
        ticket = map(lambda t:t.strip(),ticket)
        ticket = list(filter(lambda t: not t.isdigit(), ticket))
        if len(ticket) > 0:
            return ticket[0]
        else:
            return 'XXX'

    df['Ticket'] = df['Ticket'].map(cleanTicket)
    dummies_ticket = pd.get_dummies(df.Ticket,prefix='Ticket')
    df = pd.concat([df,dummies_ticket],axis=1)
    df.drop('Ticket',axis = 1,inplace = True)
    return df

combined_data = process_ticket(combined_data)

# def process_family(df):
#     df['FamilySize'] = df['Parch'] + df['SibSp'] + 1
#     df['Singleton'] = df['FamilySize'].map(lambda s: 1 if s == 1 else 0)
#     df['SmallFamily'] = df['FamilySize'].map(lambda s: 1 if 2<=s<=4 else 0)
#     df['LargeFamily'] = df['FamilySize'].map(lambda s: 1 if 5<=s else 0)
#     return df
#
# combined_data = process_family(combined_data)

def set_Age_type(df):
    df.Age = pd.cut(df.Age,[0,5,15,20,35,50,60,100])
    dummy_Age = pd.get_dummies(df.Age, prefix='Age')
    df = pd.concat([df,dummy_Age],axis=1)
    df.drop('Age',axis = 1,inplace = True)
    return df
combined_data = set_Age_type(combined_data)
combined_data.drop('PassengerId',inplace = True,axis = 1)
print(combined_data.head())


##############################################################################################################
def recover_train_test_target(df):
    train0 = pd.read_csv('train.csv')
    targets = train0['Survived']
    train = df.head(891)
    test = df.iloc[891:]
    return train,test,targets

train,test,targets = recover_train_test_target(combined_data)



from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
clf = clf.fit(train, targets)

features = pd.DataFrame()
features['feature'] = train.columns
features['importance'] = clf.feature_importances_
features.sort_values(by=['importance'], ascending=True, inplace=True)
features.set_index('feature', inplace=True)
features.plot(kind='barh', figsize=(20, 20))
#plt.show()

model = SelectFromModel(clf,threshold=0.005,prefit=True)
train_reduce = model.transform(train)
test_reduce = model.transform(test)
print(train_reduce.shape)
############################交叉验证
train_x = train_reduce[:623]
train_cv = train_reduce[623:]
train_y = targets[:623].as_matrix()
train_cv_y = targets[623:].as_matrix()

from sklearn import linear_model
from sklearn import tree
from sklearn import ensemble
from sklearn import svm
model  = ensemble.GradientBoostingClassifier(n_estimators=50)
model.fit(train_x,train_y)
print(model.score(train_cv,train_cv_y))


output = model.predict(test_reduce).astype(int)
df_output = pd.DataFrame()
aux = pd.read_csv('test.csv')
df_output['PassengerId'] = aux['PassengerId']
df_output['Survived'] = output
df_output[['PassengerId','Survived']].to_csv('output.csv',index=False)

