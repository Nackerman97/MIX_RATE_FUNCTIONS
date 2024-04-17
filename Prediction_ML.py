import pandas as pd
import numpy as np
import itertools # construct specialized tools
import matplotlib as pyplot
from matplotlib import pyplot
from sklearn.feature_selection import SelectKBest, chi2, f_classif, f_regression
from sklearn import preprocessing, ensemble, metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import learning_curve
from sklearn.metrics import roc_curve, auc, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.svm import SVC
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn import tree
#import libraries
from snowflake.sqlalchemy import URL
from sqlalchemy import create_engine
from datetime import datetime, timezone, timedelta
import datetime as dt
import random
from snowflake.connector.pandas_tools import pd_writer

PROB_PREDICTION = 0 #OR 0
TEST_SIZE = 0.80 #0.0-1.0  #0.35 for CASE
TRAIN_SIZE = 0.20
NUM_ESTIMATORS = 50 #50
LEAF_NODES = 25
USERFOLDER = 'C:/Users/nacke08/OneDrive - Safeway, Inc/Documents/Projects/Ecommerce/FY24/ML/'

#Connection setup anf functions
class SFConnection:
    def __init__(self):
        self.name = ''
        self.connection = ''
        self.engine = ''

    def open_conn(self):
        username = self.name + '@safeway.com'
        account_identifier = 'abs_itds_prd.west-us-2.privatelink'

        # Create an engine to connect to Snowflake
        self.engine = create_engine(
        f'snowflake://{username}@{account_identifier}/?authenticator=externalbrowser'
        )
        self.connection = self.engine.connect()
        print("Connection Opened")

    def test_conn(self):
        try:
            self.open_conn()
            results = self.connection.execute('select current_version()').fetchone()
            print("Snowflake connected to Version: {}".format(results[0]))
            return("CONNECTED")
        except Exception as e:
            print("Snowflake connection Error")
            return("ERR")

    def run_query(self, sql):
        self.connection.execute("BEGIN")
        self.connection.execute(sql)
        self.connection.execute("COMMIT")

    def query_to_df(self, sql):
        results = self.connection.execute(sql).fetchall()
        columns = results[0].keys()
        return pd.DataFrame(results, columns=columns)
        
    def close_conn(self):
        self.connection.close()
        print("Connection Closed")

    def add_name(self, name):
        self.name = name
    
    #accepts a dataframe and a table name in lowercase. It is fixed currently to the FBI Schema. You only need to pass through the tablename
    #The df must be in the exact format as table
    #This will upload in chuncks of 10,000 rows to Snowflake
    #This method will replace all current values from Dataframe 
    def load_dataframe_REPLACE(self, inputdf, tablename):
        inputdf.to_sql(name=tablename.lower(), con=self.engine, if_exists='replace', index=False, chunksize=10000)
    
    #Method for inserting value without replacing to data tables 
    def load_dataframe_APPEND(self, inputdf, tablename):
        inputdf.to_sql(name=tablename.lower(), con=self.engine, if_exists='append', index=False, chunksize=10000)


#TODO tree induction fitting graph to find the right number of nodes that a tree should have before overfitting. I think this is reducing 
#the perofrmance currently for case deals. This needs to be optimized 
class MachineLearning:
    def __init__(self):
        self.model = ''
        self.NumData = pd.DataFrame()
        self.Audit = pd.DataFrame()
        self.ActualData = pd.DataFrame()
        self.HHS_ID = pd.DataFrame()
        self.x_train = []
        self.x_test = []
        self.y_train = []
        self.y_test = []
        self.predicted_prob = []
        self.predicted = ''
        self.categorize_variables = []

    def set_training_data(self, data):
        #this data is the data that is used to train the database so that the new data can be tested
        dfTESTING = pd.DataFrame(data).fillna(0)
        self.categorize_variables = list(dfTESTING.iloc[:,1:-1].columns)
        dfTESTING = self.categorize_values(dfTESTING)

        #Oversample or Undersample
        ans = input("OverSample(O) or Undersample(U) or SMOTE(S) or None (N)\n")
        if ans.upper() == 'O':
            ros = RandomOverSampler(sampling_strategy='minority')
            dfTESTING, y_ros = ros.fit_resample(dfTESTING, dfTESTING.loc[:,['test_information']])
        elif ans.upper() == 'U':
            ros = RandomUnderSampler()
            dfTESTING, y_ros = ros.fit_resample(dfTESTING, dfTESTING.loc[:,['test_information']])
        elif ans.upper() == 'S':
            ros = SMOTE()
            dfTESTING, y_ros = ros.fit_resample(dfTESTING, dfTESTING.loc[:,['test_information']])

        self.Audit = dfTESTING.loc[:,['test_information']]
        self.NumData = dfTESTING.iloc[:,1:-1]
        self.NumData = self.categorize_values(self.NumData)

    def set_acutal_data(self):
        ACTUALdata = pd.read_csv (USERFOLDER + '/TESTING.csv')
        dfTESTING = pd.DataFrame(ACTUALdata).fillna(0)
        self.ActualData = dfTESTING.iloc[:,1:-1]       #remove the -1 when actual data is being tested
        self.ActualData = self.categorize_values(self.ActualData)
        self.HHS_ID = dfTESTING.loc[:,['HHS_ID']]
        #self.ActualData['HHS_ID'] = dfTESTING.loc[:,['HHS_ID']]
        #print(self.ActualData)

    def categorize_values(self, dfTemp):
        #dfData.replace(r'^\s*$', np.nan, regex=True) #replace whitespace
        for item_name in self.categorize_variables:
            dfTemp[item_name] = pd.factorize(dfTemp[item_name])[0]
        return dfTemp

    def model_selection(self):
        base_estimator = tree.DecisionTreeClassifier(criterion='gini', splitter='best')
        ans = input("SELECT WHICH MODEL TO APPLY: [1 = GradientBoost, 2 = LR, 3 = RandomForest, 4 = NaiveBayes, 5 = KNN, 6 = SVC, 7 = AdaBoost] \n")    
        if ans == "1":      self.model = ensemble.GradientBoostingClassifier(loss='log_loss', learning_rate=0.1, n_estimators=NUM_ESTIMATORS, criterion='friedman_mse')
        elif ans == "2":    self.model = LogisticRegression()
        elif ans == "3":    self.model = RandomForestClassifier(n_estimators = NUM_ESTIMATORS, max_leaf_nodes= LEAF_NODES, n_jobs = -1, random_state=42)
        #elif ans == "3":    self.model = RandomForestClassifier(n_jobs = -1, random_state=42)
        elif ans == "4":    self.model = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
        elif ans == "5":    self.model = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=1, n_neighbors=1, p=2, weights='uniform')
        elif ans == "6":    self.model = SVC(kernel='rbf', gamma='auto', probability=True)
        elif ans == "7":    self.model = ensemble.AdaBoostClassifier(base_estimator = base_estimator, n_estimators=NUM_ESTIMATORS)
        else:   exit()
    
    def preprocessing_selection(self):
        '''
        Work on integrating a technique with regards to over and under sampling. This might help make a strong case for the fewer results
        '''
        ans = input("SELECT WHICH PREPROCESSING TO COMPLETE: [1 = MIN_MAX, 2 = SS, 3 = Nomalizer, 4 = RS, 5 = HotEncoder] \n")
        if ans == "1":      preprocess = preprocessing.MinMaxScaler(feature_range = (0,1))
        elif ans == "2":    preprocess = preprocessing.StandardScaler()
        elif ans == "3":    preprocess = preprocessing.Normalizer()
        elif ans == "4":    preprocess = preprocessing.RobustScaler()
        elif ans == "5":    preprocess = preprocessing.OneHotEncoder(dtype = np.int, sparse = True)
        else:   exit()

        self.NumData = pd.DataFrame(preprocess.fit_transform(self.NumData.values),columns = self.NumData.columns)
    
    def model_fit(self):
        self.model = self.model.fit(self.x_train, self.y_train)
        

    def split_testing_data(self):

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.NumData, self.Audit, test_size = TEST_SIZE, stratify = self.Audit)

    def predict_prob(self):
        self.predicted_prob = self.model.predict_proba(self.x_test)[:,PROB_PREDICTION]
        self.predicted = self.model.predict(self.x_test)

    def plot_confusion_matrix(self, cm, classes,normalize = False, title = 'Confusion matrix', cmap = pyplot.cm.Blues):
        if normalize:
            cm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

        pyplot.imshow(cm, interpolation = 'nearest', cmap = cmap)
        pyplot.title(title, fontsize = 22)
        pyplot.colorbar()
        tick_marks = np.arange(len(classes))
        pyplot.xticks(tick_marks, classes, rotation = 45, fontsize = 13)
        pyplot.yticks(tick_marks, classes, fontsize = 13)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            pyplot.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment = 'center',
                    fontsize = 15,
                    color = 'white' if cm[i, j] > thresh else 'black')

        pyplot.tight_layout()
        pyplot.ylabel('True label', fontsize = 16)
        pyplot.xlabel('Predicted label', fontsize = 16)

    def predict_prob_actual(self):
        self.predicted_prob = self.model.predict_proba(self.ActualData)[:,PROB_PREDICTION]
        self.predicted = self.model.predict(self.ActualData)

    def write_to_csv(self):

        self.HHS_ID['PROB'] = ['%.4f' % elem for elem in self.predicted_prob]
        self.HHS_ID['PREDICT'] = ['%.4f' % elem for elem in self.predicted]

        dftesting = pd.read_csv (USERFOLDER + '/TESTING.csv')

        #output from Testing
        dfoutput = dftesting.loc[:,['HHS_ID', 'test_information']]

        df_FINAL = self.HHS_ID.merge(dfoutput[['HHS_ID', 'test_information']], on = 'HHS_ID', how = 'left')
        df_FINAL.to_csv(USERFOLDER + '/CLEANED_DATA.csv')


    def KBest_feature_selection(self):
        df = SelectKBest(f_regression, k = 5).fit_transform(self.x_train, self.y_train)
        print(df.shape)

    def dsp_learning_curve(self):
        #Must use the test train and split first before using this module from the machine leanring class
        train_sizes, train_scores, test_scores = learning_curve(estimator= self.model, X=  self.x_train, y = self.y_train, n_jobs=-1)
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        pyplot.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training Accuracy')
        pyplot.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
        pyplot.plot(train_sizes, test_mean, color='green', marker='+', markersize=5, linestyle='--', label='Validation Accuracy')
        pyplot.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
        pyplot.title('Learning Curve')
        pyplot.xlabel('Training Data Size')
        pyplot.ylabel('Model accuracy')
        pyplot.grid()
        pyplot.legend(loc='lower right')
        pyplot.show()

    def dsp_roc_curve(self):
        #Must use the test train and split first before using this module from the machine leanring class
        # generate a no skill prediction (majority class)
        ns_probs = [0 for _ in range(len(self.y_test))]

        # predict probabilities
        lr_probs = self.model.predict_proba(self.x_test)

        # keep probabilities for the positive outcome only
        lr_probs = lr_probs[:, 1]

        ns_auc = roc_auc_score(self.y_test, ns_probs)
        lr_auc = roc_auc_score(self.y_test, lr_probs)

        # summarize scores
        print('No Skill: ROC AUC=%.3f' % (ns_auc))
        print('Logistic: ROC AUC=%.3f' % (lr_auc))

        # calculate roc curves
        ns_fpr, ns_tpr, _ = roc_curve(self.y_test, ns_probs)
        lr_fpr, lr_tpr, _ = roc_curve(self.y_test, lr_probs)

        # plot the roc curve for the model
        pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
        pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
        pyplot.xlabel('False Positive Rate')
        pyplot.ylabel('True Positive Rate')
        pyplot.legend()
        pyplot.show()

    def dsp_feature_importance(self):
        # plotting feature importances
        features = self.NumData.columns
        print(self.model)
        importances = self.model.feature_importances_
        indices = np.argsort(importances)
        pyplot.figure(figsize=(10,15))
        pyplot.title('Feature Importance')
        pyplot.barh(range(len(indices)), importances[indices], color='b', align='edge')
        pyplot.yticks(range(len(indices)), [features[i] for i in indices])
        pyplot.xlabel('Relative Importance')
        pyplot.savefig(USERFOLDER +'/FEATURE_IMPORTANCE.jpeg')
        pyplot.show()
        #print([features[i] for i in indices])
        #print(importances[indices])
        #print(features)
    
    def dsp_confusion_matrix(self):
        cnf_matrix = confusion_matrix(self.y_test, self.predicted, labels = [1,0])
        np.set_printoptions(precision = 2)
        pyplot.figure()
        self.plot_confusion_matrix(cnf_matrix, classes = ['TRUE=1','FALSE=0'], normalize = False,  title = 'Confusion matrix')
        pyplot.savefig(USERFOLDER +'/confusion_matrix.png')
        pyplot.show()

    def dsp_precision_and_recall(self):
        accuracy = metrics.accuracy_score(self.y_test, self.predicted, normalize=False)
        auc = metrics.roc_auc_score(self.y_test, self.predicted_prob)
        recall = metrics.recall_score(self.y_test, self.predicted)
        precision = metrics.precision_score(self.y_test, self.predicted, zero_division=1) #this has the errro

        ############# Accuray e AUC ######################
        print("Accuracy (overall correct predictions):",  round(accuracy,2))
        print("Auc:", round(auc,2))
            
        ############# Precision & Recall #################
        print("Recall (all 1s predicted right):", round(recall,2))
        print("Precision (confidence when predicting a 1):", round(precision,2))
        print("Detail:")

        print(metrics.classification_report(self.y_test, self.predicted, target_names=[str(i) for i in np.unique(self.y_test)]))

    def dsp_max_number_of_leaves(self):
        #max_depths = np.linspace(1, 100, 100, endpoint=True)
        max_depths = [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40]
        train_results = []
        test_results = []
        for max_depth in max_depths:
            print(type(max_depth))
            rf = RandomForestClassifier(max_depth=max_depth, n_jobs=-1)
            rf.fit(self.x_train, self.y_train)
            train_pred = rf.predict(self.x_train)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(self.y_train, train_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            train_results.append(roc_auc)
            y_pred = rf.predict(self.x_test)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(self.y_test, y_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            test_results.append(roc_auc)
        from matplotlib.legend_handler import HandlerLine2D
        line1, = pyplot.plot(max_depths, train_results, 'b', label="Train AUC")
        line2, = pyplot.plot(max_depths, test_results, 'r', label="Test AUC")
        pyplot.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
        pyplot.ylabel('AUC score')
        pyplot.xlabel('Tree depth')
        pyplot.show()


def TestingMLProcess():
    ml = MachineLearning()
    ml.set_training_data()
    ml.split_testing_data()
    ml.model_selection()
    ml.preprocessing_selection()
    ml.model_fit()
    ml.predict_prob()
    ml.dsp_precision_and_recall()
    #############################
    #####ANALYSIS FUNCTIONS######
    #############################
    #ml.dsp_confusion_matrix()
    ml.dsp_feature_importance()
    #ml.dsp_roc_curve()
    #ml.dsp_max_number_of_leaves()


def test_process():
    #Pull data from Snowflake 
    #send the data through a feature importance. 
    #Start with NPS to determine if the customer came back in 30 60 or 90 days 
    SQL = '''
    select 
    mrs.TXN_ID,
    previous_store_sales_category,
    mrs.DELIVERY_TYPE, 
    mrs.SVCLVL_TYPE,
    mrs.DIVISION,
    mrs.URBANICITY,
    mrs.OM_ID,
    mrs.DISTRICT,
    mrs.FACTS,
    CASE WHEN FRST_TXN_SAVE20_PROMO_CODE_FLAG > 0 or FRST_TXN_SAVE30_PROMO_CODE_FLAG > 0 OR CYBER_CODE > 0 THEN 1 ELSE 0 end as PROMO_USE,
    case when repeat_60_days_ecomm > 0 then 1 ELSE 0 END AS TEST_INFORMATION
    from "EDM_BIZOPS_PRD"."FBI"."ESA_MIX_RATE_SEGMENTATION" mrs
    left join "EDM_BIZOPS_PRD"."FBI"."ESA_HOLISTIC_SCORECARD_TXN_CUSTOMER_SEGMENT" cs
    on mrs.txn_id = cs.txn_id
    WHERE 
    cs.segment_1 IN ('BNC TO ACI', 'BNC TO ECOMM')
    AND FISCAL_QUARTER_ID = '202304'
    '''

    conn = SFConnection()
    conn.add_name('NACKE08')
    conn.test_conn()
    dfRESULTS = conn.query_to_df(SQL)
    conn.close_conn()

    ml = MachineLearning()
    ml.set_training_data(dfRESULTS)
    ml.split_testing_data()
    ml.model_selection()
    ml.preprocessing_selection()
    ml.model_fit()
    ml.predict_prob()
    ml.dsp_precision_and_recall()
    #############################N
    #####ANALYSIS FUNCTIONS######
    #############################
    ml.dsp_confusion_matrix()
    ml.dsp_feature_importance()
    #ml.dsp_roc_curve()
    #ml.dsp_max_number_of_leaves()


"""
DATE: 11/23/2021
PURPOSE:
REFERENCES: TestingMLProcess, ActualMLProcess
"""
def main():
    print("Start of Program")
    #TestingMLProcess()
    test_process()


    print("End of Program")

if __name__ == "__main__":
    main()

    '''
    NOTES:

    Oversampling and undersampling
    --Both can help with imbalanced datasets. 

    Oversampling takes the minority class and increases the likelihood of events. This is beneficial, but can lead to overfitting 

    Undersampling reduced the majority class. This can also balance the data, but can remove potenital useful cases from the model


    Synthetic Minority Over-sampling Technique: Mitigrates the problem of overfittings caused by random oversampling and syntetic examples a
    re generated rather than replicated of instances. No loss of useful information --- Not effective for high dimenstional data


    Data Science Process
    -Two methods for the data collection. I can collect all of the data from a column level were each of the values is categorized. 
    I could also pull all data from a item level and label 1 and 0 for True and False. Both options are valid and can be used for different purposes. 
    The first method can produce the churn model and highlight overall feature importance while the second method can also predict churn but requires 
    a larger amount of testing an training data, but can idenitify the specific characteritics that is leading to churn. 

    -The next step in the process is to try a variety of different data sciennce models. These will help produce a guage for which ones might be overfitting or underfitting the
    data. This can also be determined by looking at the testing data. We will want to make sure that this is a reasonable amount of the chruned values in the dataset or else the overfitting 
    will occur. Therefore we will need to use the overfitting boost to help the model out. 

    -The next step would be to identify the number of root nodes needed by the mdoel. What is the Tree size and the Forest size if a Random Forest classifier model is used. This can be viewed 
    by running the test data through a AUC model. 

    -Naive Bayes and Calssifiation Tree are going to be better for Churn Prediciton 

    -

    '''