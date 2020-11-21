#Objetivo: Hacer un programa para automatizar la limpieza de las bases de
#datos dada.

#Librerias que se van a utilizar
#Primeros paquetes
import pandas as pd #importaremos panda para leer los arvhivos csv
import numpy as np

#Estos paquetes son para graficar y no los utilizare en este momento
#import matplotlib.pyplot as plt
#import seaborn as sns
#color = sns.color_palette
#sns.set_style('darkgrid')

from scipy import stats
from scipy.stats import norm, skew

#Tercer paquete
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest

#imputation packs
from sklearn import neighbors #para 
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer, KNNImputer #imputar NaN data

#Cuarto paquete
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression

print('We want to read the data. White the file with extension .csv')
train = input('File name for training data:\n')
if len(train) < 1 : train = "train.csv"
test = input('File name for test data:\n')
if len(test) < 1 : test = "test.csv"

try:
    predi = input('Write the name of the prediction feature.\n')
    Id = input('Column that label the dataframe.\n')
        #predi = srt(pred)
    dbtrainid = pd.read_csv(train , sep=',',header='infer')
    train_ID = dbtrainid[Id]
    objetivo = np.log1p(dbtrainid[predi])
    #objetivo = dbtrainid[predi]
    dbtrain = dbtrainid.drop([Id,predi],axis=1)
    dbtestid = pd.read_csv(test ,sep=',',header='infer')
    dbtest = dbtestid.drop(Id,1)
    test_ID = dbtestid[Id]
    index = pd.Series(test_ID).array
    dbunion = pd.concat([dbtrain,dbtest])
except:
    print('Can not found the file')
    quit()

delete = input('Do you want to delete any column?, [name of column/ n]\n')
while True:
        if delete == 'n':
            print('Any column deleted.')
            break
        else:
            try:
                dbunion = dbunion.drop([delete],axis=1)
                print('The column', delete, 'is deleted.')
                nuevo = input('Do you want do erase another column? [y/n]:\n')
                if nuevo == 'n':
                    print('Ok.')
                    break
                if nuevo == 'y':
                    delete = input('Which one?\n')
                else:
                    print('That is not an answer, write y or n.') 
                    nuevo = input('Do you want do erase another column? [y/n]:\n')
            except:
                print('Have not found the column', delete,'.Try again.')
                delete = input('Do you want to delete any column?, [name of column/ n]\n')

print('Your train data has', dbtrain.shape[0], 'rows and has', dbtrain.shape[1], 'columns.')
print('Your test data has', dbtest.shape[0], 'rows and has', dbtest.shape[1], 'columns .')

union_na = (dbunion.isnull().sum() / len(dbunion)) * 100
union_na2 = union_na.drop(union_na[union_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :union_na2})
print('The ratio for missing data by columns of the two data bases is:')
print(missing_data)


print('The next question will allow to transform strings values into integrers (one integrer per parameter).If your answer is No all missing data will become zero.')
resp = input('Do you like to index every parameter in columns?. [y/n]:\n',)

while True:
    if resp == 'y':
        le = preprocessing.LabelEncoder()
        prep = dbunion.dtypes.eq(np.object)
        dbunion.loc[:, prep] = dbunion.loc[:, prep].astype(str).apply(le.fit_transform)
        gauss = input('Do you want to standardize your data?. This means, standardize features by removing the mean and scaling to unit variance. [y/n]:\n')
        if gauss == 'y':
            scaler = StandardScaler()
            c = []
            for i in range (0,int(dbunion.shape[1])):
                c.append(dbunion.iloc[i][0])
                arr = np.asarray(c).reshape(-1, 1)
                #print(arr)
                #arr.loc[:,:] = arr.apply(scaler.fit_transform)
                scaler.fit(arr)
        else:
            print('No change made.')
        union_na = (dbunion.isnull().sum() / len(dbunion)) * 100
        union_na2 = union_na.drop(union_na[union_na == 0].index).sort_values(ascending=False)
        missing_data = pd.DataFrame({'Missing Ratio' :union_na2})
        print('The ratio for missing data by columns is:')
        print(missing_data)
        break
    elif resp == 'n':
        dbunion = dbunion.fillna(0)
        le = preprocessing.LabelEncoder()
        prep = dbunion.dtypes.eq(np.object)
        dbunion.loc[:, prep] = dbunion.loc[:, prep].astype(str).apply(le.fit_transform)
        union_na = (dbunion.isnull().sum() / len(dbunion)) * 100
        union_na2 = union_na.drop(union_na[union_na == 0].index).sort_values(ascending=False)
        missing_data = pd.DataFrame({'Missing Ratio' :union_na2})
        #print('The ratio for missing data by columns is:')
        #print(missing_data)
        break
    else:
        print('Just write y or n.')
        resp = input('Would you like to index every parameter in columns?. [y,n]:\n',)

if len(missing_data)<1:
    print('Ok, going to regression models.')
else:
    choice1 = input('What to do with remaining data? \n 1) KNN imputation \n 2) Mean imputation \n 3) Do it all zero \n 4) Nothing \n',)
    while True:
        if choice1 == '1':
            #classifier= neighbors.KNeighborsClassifier(n_neighbors=5) #Classifier implementing the k-nearest neighbors vote.
            #neigh = NearestNeighbors(n_neighbors=5)
            #classifier.fit(dbunion,objetivo)
            #Lo que necesitamos es imputar datos a los NaN a traves de las vecindades K cercanas
            vec = input('Number of neighbors to use:\n ')
            while True:
                try:
                    veci = int(vec)
                    imputer = KNNImputer(n_neighbors=veci, weights='uniform', metric='nan_euclidean') #define el imputador
                    #imputer.fit(dbunion)
                    dbunion = pd.DataFrame(imputer.fit_transform(dbunion) , columns = dbunion.columns)
                    break
                except:
                    print('Write an integrer.')
                    vec = input('Number of neighbors to use:\n ')
            union_na = (dbunion.isnull().sum() / len(dbunion)) * 100
            union_na2 = union_na.drop(union_na[union_na == 0].index).sort_values(ascending=False)
            missing_data = pd.DataFrame({'Missing Ratio' :union_na2})
            break
        if choice1 == '2':
            #classifier= neighbors.KNeighborsClassifier(n_neighbors=5) #Classifier implementing the k-nearest neighbors vote.
            #neigh = NearestNeighbors(n_neighbors=5)
            #classifier.fit(dbunion,objetivo)
            #Lo que necesitamos es imputar datos a los NaN a traves de las vecindades K cercanas
            imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
                            #add_indicator=True) #define el imputador
                    #imputer.fit(dbunion)
            dbunion = pd.DataFrame(imputer.fit_transform(dbunion) , columns = dbunion.columns)
            union_na = (dbunion.isnull().sum() / len(dbunion)) * 100
            union_na2 = union_na.drop(union_na[union_na == 0].index).sort_values(ascending=False)
            missing_data = pd.DataFrame({'Missing Ratio' :union_na2})
            break
        if choice1 == '3':
            dbunion = dbunion.fillna(0)
            union_na = (dbunion.isnull().sum() / len(dbunion)) * 100
            union_na2 = union_na.drop(union_na[union_na == 0].index).sort_values(ascending=False)
            missing_data = pd.DataFrame({'Missing Ratio' :union_na2})
            break
        if choice1 == '4':
            union_na = (dbunion.isnull().sum() / len(dbunion)) * 100
            union_na2 = union_na.drop(union_na[union_na == 0].index).sort_values(ascending=False)
            missing_data = pd.DataFrame({'Missing Ratio' :union_na2})
            break
        else:
            print('Just write y or n.')
            choice1 = input('Do you like to make zero all the missing data remaining. [y,n] \n',)

#Outliers
if len(missing_data)<1:
    print('There is no missing data left.')
    outliers = input('Do you want to study the outliers?. [y,n]\n')
    if outliers == 'y':
        # identify outliers in the training dataset
        iso = IsolationForest(contamination=0.1, n_estimators=100,max_features=1)
        for columns in dbunion:
            columna = dbunion.loc[:, columns].to_numpy().reshape(-1, 1)
            iso.fit(columna,objetivo)
            dbunioncol = iso.fit_predict(columna,objetivo)
            ser = pd.Series(dbunioncol).map({1:0,-1:1})
            print(columns,'\n',ser.value_counts())
            #try:
             #   if ser.value_counts()[1]<50:
              #      print(columns ,'\n', ser.value_counts())
            #except:
             #   continue
        #for columns in dbunion:
         #   columna = dbunion.loc[:, columns].to_numpy().reshape(-1, 1)
          #  iforest = iso.fit(columna)
           # dbunioncol = iso.predict(columna)
            #dbunion = pd.DataFrame(dbunioncol)
            #break
        #dbunion = iso.fit_predict(dbunion)
        #dbunion = pd.DataFrame(iso.fit_predict(dbunion) , columns = dbunion.columns)
        print(len(dbunion))
        print('Outliers are gone.')
        #mask = yhat != -1
        #dbtrain, objetivo = dbtrain[mask, :], objetivo[mask]
    else:
        print('The outliers still.')
else:
    print('There are', len(missing_data),' columns with data missing remaining. The ratio of the missing data stills:')
    print(missing_data)
    print('You cant do a regression model.')
    quit()

#CleanedData
cleand = input('Do you want to save a csv of your cleaned data?, [y/n]:\n')
while True:
    if cleand == 'y':
        trainclean = dbunion.iloc[:int(dbtrain.shape[0]),:]
        testclean = dbunion.iloc[int(dbtrain.shape[0]):,:]
        trainclean.to_csv('traincleanData.csv')
        testclean.to_csv('testcleanData.csv')
        break
    if cleand == 'n':
        trainclean = dbunion.iloc[:int(dbtrain.shape[0]),:]
        testclean = dbunion.iloc[int(dbtrain.shape[0]):,:]
        break
    else:
        print('Just weite y or n.')
        cleand = input('Do you want to save a csv of your cleaned data?, [y/n]:\n')


#Regression models
regre = input('Want to do a regression model?, [y/n]: \n')
while True:
    if regre == 'y':
        mod = input('What kind of regression do you want? \n Linear Regession, Lasso Regression or Ridge Regression? \n write [lr/lm/rm]:\n')
        while True:
            if mod == 'lr':
                lr = LinearRegression(normalize=True)
                lr.fit(trainclean, objetivo)
                salida1 = pd.DataFrame(lr.predict(testclean), index=index)
                salida1[predi] = np.exp(salida1[0])
                salida1 = salida1.drop([0],axis='columns')
                salida1.rename(columns={'':'Id'},
                         inplace=True)
                print(salida1)
                res1 = input('Do you want a csv file for this results? [y,n]\n')
                if res1 == 'y':
                    nom1 = input('Name the file with .csv extension:\n')
                    salida1.to_csv(nom1)
                else:
                    print('The file will not be generated.')
                again = input('Do you want to chose another model? [y/n]\n')
                if again =='y':
                        mod = input('What kind of regression do you want? \n Linear Regession, Lasso Regression or Ridge Regression? \n write [lr/lm/rm]:\n')
                else:
                    print('Ok. Bye.')
                    break
            elif mod == 'lm':
                alphalm = input('What alpha parameter do you want?\n')
                while True:
                    try:
                        alph1 = float(alphalm)
                        print('The Lasso model with parameter',alph1,'is:')
                        break
                    except:
                        print('The alpha must be a number.')
                        alphalm = input('What alpha parameter do you want?\n')    
                lm = Lasso(alpha=float(alphalm), max_iter=1e6)
                lm.fit(trainclean, objetivo)
                salida2 = pd.DataFrame(lm.predict(testclean),index=index)
                salida2[predi]= np.exp(salida2[0])
                salida2 = salida2.drop([0],axis='columns')
                salida2.rename(columns={'':'Id'},
                         inplace=True)
                print(salida2)
                res2 = input('Do you want a csv file for this results? [y,n]\n')
                if res2 == 'y':
                    nom2 = input('Name the file with .csv extension:\n')
                    salida2.to_csv(nom2)
                else:
                    print('The file will not be generated.')
                again = input('Do you want to chose another model? [y/n]\n')
                if again =='y':
                        mod = input('What kind of regression do you want? \n Linear Regession, Lasso Regression or Ridge Regression? \n write [lr/lm/rm]:\n')
                else:
                    print('Ok. Bye.')
                    break
            elif mod == 'rm':
                alpharm = input('What alpha parameter do you want?\n')
                while True:
                    try:
                        alph2 = float(alpharm)
                        print('The Ridge model with parameter',float(alpharm),'is:')
                        break
                    except:
                        print('The alpha must be a number.')
                        alpharm = input('What alpha parameter do you want?\n')
                rm = Ridge(alpha=float(alpharm))
                rm.fit(trainclean, objetivo)
                salida3 = pd.DataFrame(rm.predict(testclean),index=index)
                salida3[predi] = np.exp(salida3[0])
                salida3 = salida3.drop([0],axis='columns')
                salida3.rename(columns={'Unamed=0':'Id'},
                         inplace=True)
                print(salida3)
                res3 = input('Do you want a csv file for this results? [y,n]\n')
                if res3 == 'y':
                    nom3 = input('Name the file with .csv extension:\n')
                    salida3.to_csv(nom3)
                else:
                    print('The file will not be generated.')
                again = input('Do you want to chose another model? [y/n]\n')
                if again =='y':
                        mod = input('What kind of regression do you want? \n Linear Regession, Lasso Regression or Ridge Regression? \n write [lr/lm/rm]:\n')
                else:
                    print('Ok. Bye.')
                    break
            else:
                print('That is not a correct option.')
                mod = input('What kind of regression do you want? \n Linear Regession, Lasso Regression or Ridge Regression? \n write [lr/lm/rm]:\n')
    if regre == 'n':
        print('Ok. Bye.')
        break
    else:
        print('Just write y or n.')
        regre = intput('Want to do a regression model?, [y/n]: \n')
