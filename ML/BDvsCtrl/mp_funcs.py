import numpy as np
import pandas as pd
import scipy
import pandas as pd
from sklearn.model_selection import ShuffleSplit
import pickle

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
    
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline     
    
    
    
def process_pval(df, csv_fname= None, n_splits=50, random_state= 123): 
    print(csv_fname)
    stats_pval = np.empty( (0, len(df.columns)))
    
    ss = ShuffleSplit(n_splits=n_splits, test_size=0.5, random_state=random_state)
    for train_index_, _ in ss.split(df):
        X = df.iloc[train_index_,:] # take subset as stat calculation
        # calc p-value for each train dataset between two condition group
        stat_result = scipy.stats.mannwhitneyu(X[X['condition']==0] , X[X['condition']==1] )
        stats_pval = np.vstack((stats_pval, stat_result[1]))
        
    pval_df = pd.DataFrame(stats_pval, columns=df.columns)
    pval_df.drop(columns=['condition'], inplace=True)
    if not csv_fname: 
        pval_df.to_csv(csv_fname)
    sortIdx = np.argsort(pval_df.median())
    # featuresList =list(pval_df.iloc[:, sortIdx[:500]].columns)
    featuresList =pval_df.iloc[:, sortIdx[:500]]
    return featuresList
    
    
def feature_selection(data, numFolds, result_dir, test_size = 10): 
    featuresPerIter =[]               
    index_tables=[]

    ss = ShuffleSplit(n_splits=numFolds, test_size=test_size, random_state=123)
    for i, (train_index, test_index) in enumerate(ss.split(data)):
        print(f"Fold {i}:")
        index_tables.append({'fold':i, 'train_index': train_index, 'test_index':test_index})

        
        train_dataset = data.iloc[train_index,:]

        stats_pval = np.empty( (0, len(train_dataset.columns)))
        ss = ShuffleSplit(n_splits=numFolds, test_size=0.5)
        for train_index_, _ in ss.split(train_dataset):
            X = train_dataset.iloc[train_index_,:] # take subset as stat calculation
            stat_result = scipy.stats.mannwhitneyu(X[X['condition']==0] , X[X['condition']==1] )
            stats_pval = np.vstack((stats_pval, stat_result[1]))

        pval_df = pd.DataFrame(stats_pval, columns=data.columns)    
        pval_df.to_csv(f'{result_dir}/pval_df_{i}.csv')


        sortIdx = np.argsort(pval_df.median())
        featuresList = list(pval_df.median()[sortIdx[:500]].index)
        featuresList.remove('condition')
        featuresPerIter.append(featuresList)    


    with open(f'{result_dir}/featuresPerIter.pk', 'wb') as f: 
        pickle.dump( featuresPerIter, f)    
    with open(f'{result_dir}/index_tables.pk', 'wb') as f: 
        pickle.dump( index_tables, f)
    print('Done')
    
# Define the classifiers to compare
classifiers = [
    ('Logistic Regression', LogisticRegression(random_state=42, max_iter =1000)),
    # ('Decision Tree', DecisionTreeClassifier(random_state=42)),
    ('Random Forest', RandomForestClassifier(random_state=42)),
    # ('Gradient Boosting', GradientBoostingClassifier(random_state=42)),
    ('K-Nearest Neighbors', KNeighborsClassifier()),
    # ('Linear SVM', LinearSVC(random_state=42, probabilities=True)),
    ('SVM', SVC(random_state=42)),
    ('Neural Net',  MLPClassifier(max_iter = 1000, activation='relu', solver = 'lbfgs', learning_rate = 'constant', 
                                  alpha=0.0001, hidden_layer_sizes=(10, 30, 10), verbose=False,)),
]


from sklearn.model_selection import ShuffleSplit
def global_predicat(data, features): 
    score=[]
    ss = ShuffleSplit(n_splits=20, test_size=0.5, random_state=None)
    for train_index, test_index in ss.split(range(data.shape[0])):
        clf = LogisticRegression(random_state=42, max_iter =1000)

        X_train = np.array(data.iloc[train_index,:].loc[:, features])
        y_train= np.array(data.iloc[train_index,:].loc[:,'condition'])      

        X_test = np.array(data.iloc[test_index,:].loc[:, features])
        y_test= np.array(data.iloc[test_index,:].loc[:,'condition'])    

        pipe = Pipeline([ ('scaler', StandardScaler()),  ('clf', clf),  ]) 
        pipe.fit(X_train,y_train)
        score.append(pipe.score(X_test,y_test))        
        
    return np.mean(score)
