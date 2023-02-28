import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from columndescription import Description
from sklearn.naive_bayes import MultinomialNB

z = Description()
# IMPORT DATA........................

data = pd.read_csv("path/to/file")  # path of data file
# print (data)


# EXPLORE DATASET.......................

#print(data.shape)
print(data.columns)
print(data.info())

data = data.drop(["NAME","EMAIL_ID","REGISTRATION_IPV4"],axis=1)
# print(data)

new_data=data.dropna(subset=['ISBOT'])
# print(new_data)

# print(new_data.columns)
# print(new_data.info())
print(new_data.shape)


all_null = new_data .isnull().sum()
# print (all_null)

pd.set_option('mode.chained_assignment',None)

categorical = [col for col in new_data.columns if new_data[col].dtypes == 'O']
# print("categorical colums are.....",categorical)

numerical = [col for col in new_data.columns if new_data[col].dtypes != 'O']
# print("numerical colums are.....",numerical)

for cat in categorical:
    print(cat)
    z.category_columns(new_data,cat)


# print("for loop ended =====>>>")

for num in numerical:
    print(num)
    z.numerical_columns(new_data,num)

# # WORKING ON GENDER


new_data.GENDER = [1 if i == "Male" else 0 for i in new_data.GENDER]
# print(new_data.GENDER)

# # # WORKING ON IS_GLOGIN

new_data.IS_GLOGIN = [1 if i == True else 0 for i in new_data.IS_GLOGIN]
# print(new_data.IS_GLOGIN)


# # # WORKING ON REGISTRATION_LOCATION
all_names = new_data['REGISTRATION_LOCATION'].unique()
# print("ALL NAMES =>>",all_names)
dic = dict((value,index) for index,value in enumerate(all_names))
# print("dict from all name =>",dic)

new_data['REGISTRATION_LOCATION'] = new_data['REGISTRATION_LOCATION'].map(dic)
# print(new_data['REGISTRATION_LOCATION'].tail(1000))
# print(new_data['REGISTRATION_LOCATION'].head(10000))

# # # print(new_data.info())

# # working on ISBOT..................
new_data.ISBOT = [1 if i == True else 0 for i in new_data.ISBOT]
# # print((new_data.ISBOT).unique())
# # print((new_data.ISBOT))

# # # ....................TRAINING AND TEST DATA SPLITTING......................

x_data = new_data.drop(["ISBOT"],axis = 1)
# print(x_data)

# # c = x_data .isnull().sum()
# # # print(c)

y_data = new_data.ISBOT.values
# print(y_data)
 

arr = np.array(y_data)
y = np.where(arr == 1)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.3, random_state=1)


# # # ..........applying guassian naive bayes.................


gnb = GaussianNB()

gnb.fit(x_train, y_train)

# # Checking training and testing accuracy.......

print("print Train for accuracy of GNBC algo: ", gnb.score(x_train,y_train))
print("print Test for accuracy of GNBC algo: ", gnb.score(x_test,y_test))



# # # checking model accuracy.......

y_pred = gnb.predict(x_test)

print(y_pred)

print("true labelled bot account...",np.count_nonzero(y_pred == 1))

print("false labelled bot account",np.count_nonzero(y_pred == 0))

print('GNB Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))



# # # # ...........CONFUSION MATRIX.....................

cm = confusion_matrix(y_test, y_pred)

print('Confusion matrix\n\n', cm)

print('\nTrue Positives(TP) = ', cm[1,1])

print('\nTrue Negatives(TN) = ', cm[0,0])

print('\nFalse Positives(FP) = ', cm[0,1])

print('\nFalse Negatives(FN) = ', cm[1,0])


print("GNB CLASSIFICATION REPORT \n",classification_report(y_test, y_pred))

TN = cm[0,0]
TP = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]

classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)

print('GNB Classification accuracy : {0:0.4f}'.format(classification_accuracy))

classification_error = (FP + FN) / float(TP + TN + FP + FN)

print('GNB Classification error : {0:0.4f}'.format(classification_error))





# # # # # .............Applying multinomial naive bayes.............



mnb = MultinomialNB(alpha=1)

mnb.fit(x_train, y_train)

# # # # # CHECKING training and testing accuracy...............

print("print Train for accuracy of MNBC algo: ", mnb.score(x_train,y_train))
print("print Test for accuracy of MNBC algo: ", mnb.score(x_test,y_test))


# # # # # checking model accuracy.......

y_pred = mnb.predict(x_test)

print(y_pred)
print("true labelled bot account...",np.count_nonzero(y_pred == 1))

print("false labelled bot account...",np.count_nonzero(y_pred == 0))

print('MNB Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


# # # # # ...........CONFUSION MATRIX.....................

cm = confusion_matrix(y_test, y_pred)

print('Confusion matrix\n\n', cm)

print('\nTrue Positives(TP) = ', cm[1,1])

print('\nTrue Negatives(TN) = ', cm[0,0])

print('\nFalse Positives(FP) = ', cm[0,1])

print('\nFalse Negatives(FN) = ', cm[1,0])


print("MNB CLASSIFICATION REPORT \n",classification_report(y_test, y_pred))

TN = cm[0,0]
TP = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]

classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)

print('MNB Classification accuracy : {0:0.4f}'.format(classification_accuracy))

classification_error = (FP + FN) / float(TP + TN + FP + FN)

print('MNB Classification error : {0:0.4f}'.format(classification_error))



