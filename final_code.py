import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import category_encoders as ce
from scipy.stats import pearsonr
from scipy import stats
from sklearn.impute import KNNImputer
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from xgboost.sklearn import XGBRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_regression
from pca import pca

class SalaryPrediction:
    
    def inputData(self, file1, file2, file3):    
        self.train_features = pd.read_csv("train_features.csv")
        self.train_salaries = pd.read_csv("train_salaries.csv")
        self.test_features = pd.read_csv("test_features.csv")

    def plotAndSave(self, feature_name):
        pcount = sns.countplot(x=feature_name,data=train_features)
        plt.show()
        plt.savefig(feature_name+"distribution.png")


    def exploreData(self):
        print(self.train_features.head())
        print(self.train_features.info())
        #Can find the columns, counts, type i.e. continuous/discrete and number of nans
        print(self.train_salaries.head())
        self.train_features = pd.merge(self.train_features,self.train_salaries,on="jobId")
        self.plotAndSave("degree")
        self.plotAndSave("major")
        self.plotAndSave("jobType")
        self.plotAndSave("companyId")
        #Since the distributions are quite equal and no skewness can be used to fill missing data
        for category in ['degree','major','jobType','industry','companyId']:
            print(len(self.train_features[category].unique()))

    def cleanData(self):
        self.train_features.replace(to_replace='NONE', value=np.nan, inplace=True)
        

        


#test_features = pd.read_csv("test_features.csv")


print(train_features.degree.unique())


print(train_features.info())
'''


#It can be seen data is evenly distributed, no skewness to fill missing values

'''
for category in ['degree','major','jobType','industry','companyId']:
    print(len(train_features[category].unique()))

#for less and meaningful categories convert to binaryencoder
encoder = ce.BinaryEncoder(cols=['degree','major','jobType','industry','companyId'])
train_features_binary = encoder.fit_transform(train_features)

test_features_binary = encoder.fit_transform(test_features)
print(train_features.iloc[0])


train, test = train_test_split(train_features, train_size=0.75, test_size=0.25, random_state=42, shuffle=True)
X_train = train.iloc[:,[1,2,3,4,5]]
y_train = train.iloc[:,-1]
X_test = test.iloc[:,[1,2,3,4,5]]
'''
fs = SelectKBest(score_func=mutual_info_regression, k='all')
# learn relationship from training data
fs.fit(X_train, y_train)
# transform train input data
X_train_fs = fs.transform(X_train)
# transform test input data
X_test_fs = fs.transform(X_test)
for i in range(len(fs.scores_)):
	print('Feature %d: %f' % (i, fs.scores_[i]))
# plot the scores
pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
pyplot.show()





model = pca()
# Fit transform
train_drop = train_features_binary.drop(columns=["jobId"])
out = model.fit_transform(train_drop)
'''
null_data = train_features[pd.isnull(train_features["major"]) | pd.isnull(train_features["degree"])]
matrix =train_features.corr(method='pearson')
sns.heatmap(matrix)
plt.show()
'''
degree_pcount = sns.countplot(x="degree",data=null_data)
plt.show()
plt.savefig("degree_distribution_null.png")

major_pcount = sns.countplot(x="major",data=null_data)
plt.show()
plt.savefig("major_distribution_null.png")


jobtype_pcount = sns.countplot(x="jobType",data=null_data)
plt.show()
plt.savefig("jobtype_distribution_null.png")

company_pcount = sns.countplot(x="companyId",data=null_data)
plt.show()
plt.savefig("company_distribution_null.png")

'''
percentange_major_null = len(train_features[pd.isnull(train_features["major"])])/len(train_features)
percentage_degree_null = len(train_features[pd.isnull(train_features["degree"])])/len(train_features)

major_not_null = train_features[~pd.isnull(train_features["major"])]
degree_not_null = train_features[~pd.isnull(train_features["degree"])]

print(percentange_major_null, percentage_degree_null)
#major 50% missing, so can drop the column
#before deleting find correlation to salary
'''
from dython import nominal
nominal.associations(train_features,figsize=(20,10),mark_columns=True);
'''
values = []
for industry in train_features.industry.unique():
    _df = train_features[train_features.industry==industry]
    values.append(_df.salary.values)

# compute the ANOVA
# with starred *list
# as arguments
print("industry",stats.f_oneway(*values))

values = []
for company in train_features.companyId.unique():
    _df = train_features[train_features.companyId==company]
    values.append(_df.salary.values)

# compute the ANOVA
# with starred *list
# as arguments
print("company",stats.f_oneway(*values))

values = []
for jobType in train_features.jobType.unique():
    _df = train_features[train_features.jobType==jobType]
    values.append(_df.salary.values)

# compute the ANOVA
# with starred *list
# as arguments
print("jobType",stats.f_oneway(*values))

# Print the top features. The results show that f1 is best, followed by f2 etc



train_features_binary.drop(columns=["major_0","major_1","major_2","major_3","degree_0","degree_1","degree_2"])
null_data = null_data[pd.isnull(null_data["degree"])] 

null_data = null_data[pd.isnull(null_data["degree"])] 

not_null_df = train_features[~pd.isnull(train_features["major"])]

values = []
for major in major_not_null.major.unique():
    _df = major_not_null[major_not_null.major==major]
    values.append(_df.salary.values)
print("major",stats.f_oneway(*values))
print(stats.f.ppf(q=1-0.05, dfn=5, dfd=len(major_not_null)-1))
values = []
for degree in degree_not_null.degree.unique():
    _df = degree_not_null[degree_not_null.degree==degree]
    values.append(_df.salary.values)

# compute the ANOVA
# with starred *list
# as arguments
print("degree",stats.f_oneway(*values))
print(stats.f.ppf(q=1-0.05, dfn=3, dfd=len(degree_not_null)-1))
print(not_null_df["major"].unique())
F, p = stats.f_oneway(not_null_df[not_null_df.major=='MATH'].salary,
                      not_null_df[not_null_df.major=='PHYSICS'].salary, 
                      not_null_df[not_null_df.major=='CHEMISTRY'].salary,
                      not_null_df[not_null_df.major=='BIOLOGY'].salary,
                      not_null_df[not_null_df.major=='LITERATURE'].salary,
                     not_null_df[not_null_df.major=='ENGINEERING'].salary,
                      not_null_df[not_null_df.major=='BUSINESS'].salary)

print(F)

not_null_df = train_features[~pd.isnull(train_features["degree"])]
print(not_null_df["degree"].unique())
F, p = stats.f_oneway(not_null_df[not_null_df.degree=='MASTERS'].salary,
                      not_null_df[not_null_df.degree=='HIGH_SCHOOL'].salary,
                      not_null_df[not_null_df.degree=='DOCTORAL'].salary,
                      not_null_df[not_null_df.degree=='BACHELORS'].salary,)

print(F)

jobtype_pcount = sns.countplot(x="jobType",data=null_data)
plt.show()
plt.savefig("jobtype_distribution_null.png")

#most in janitor is missing
#both features are important, so we have to impute them
imputer = KNNImputer(n_neighbors=3)
print(train_features_binary.columns)
train_features_binary = train_features_binary.drop(columns=["jobId"])
imputed_data = imputer.fit_transform(train_features_binary)
imputed_df = pd.DataFrame(imputed_data, columns=train_features_binary.columns)
print(imputed_df.head())


df_continuous = imputed_df[['yearsExperience', 'milesFromMetropolis','salary']]

sns.boxplot(data=df_continuous, orient="h")
plt.show()
plt.savefig("outliers.png")

#target variable salary has outliers using box and whisker plot, bring to bounds from box whisker plot
lower_salary = 17
upper_salary = 219
imputed_df.loc[imputed_df['salary'] > upper_salary, 'salary'] = upper_salary
imputed_df.loc[imputed_df['salary'] < lower_salary, 'salary'] = lower_salary

scaler = StandardScaler()
scaled_values = scaler.fit_transform(imputed_df.values)
scaled_features = pd.DataFrame(scaled_values, index = imputed_df.index, columns = imputed_df.columns)

test_features_binary = test_features_binary.drop(columns=["jobId"])
scaled_test = scaler.fit_transform(test_features_binary)
scaled_test_features = pd.DataFrame(scaled_test, index = imputed_df.index, columns = imputed_df.columns)

train, test = train_test_split(scaled_features, train_size=0.75, test_size=0.25, random_state=42, shuffle=True)
X_train = train.iloc[:,:-1]
y_train = train.iloc[:,-1]


rfr = RFR(n_estimators=10, random_state=0)
rfr.fit(X_train,y_train)
y_pred = rfr.predict(X_train)
print('Train accuracy score:',r2_score(y_train,y_pred))

'''
models = [('lr',LinearRegression()),('rfr',RFR(n_estimators=30, random_state=0))]
stacking = StackingRegressor(estimators=models)
#svr = LinearRegression()
'''
params = {'rfr__max_depth': [3,4,5], 'rfr__min_samples_split': [2,4], 'rfr__max_features': [4,6,8] }

#grid = GridSearchCV(estimator=stacking, param_grid=params, cv=5)
#grid.fit(X, y)
'''
grid.fit(X_train,y_train)
y_pred = grid.predict(X_train)
print('Train accuracy score:',r2_score(y_train,y_pred))
'''

print(scaled_features.columns)
lr = LinearRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
print(sklearn.metrics.mean_squared_error(y_train,y_pred))
print('Train accuracy score:',r2_score(y_train,y_pred))
#need target to be of int

y_test_pred = lr.predict(scaled_test_features)
#sqrt(mean((test_data$Sales - predict(fit_0, test_data)) ^ 2))

