import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import pandas as pd
import numpy as np
import os
#test
import scipy.stats as stats

# Viz imports
import matplotlib.pyplot as plt
import seaborn as sns

# Modeling imports
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings("ignore")
α = .05
alpha= .05
from sklearn.impute import SimpleImputer
#image
from IPython.display import Image
from IPython.core.display import HTML 

#picture-----------------------------------------------------------------------------------------------------------------------#-------------------------------------------------------------------------------------------------------------------------------#------------------------------------------------------------------------------------------------------------------------------ 
def imagine1():
    '''importing a imagine from the web'''
    return Image(url= "https://www.paymentscardsandmobile.com/wp-content/uploads/2018/08/Future-of-banking.jpg", width=1000, height=1000)






#acquire-----------------------------------------------------------------------------------------------------------------------#-------------------------------------------------------------------------------------------------------------------------------#------------------------------------------------------------------------------------------------------------------------------ 

def acquire_banking():
    '''acquire file from local csv downloaded from kaggle'''
    df = pd.read_csv('banking.csv')
    return df




#prepare-----------------------------------------------------------------------------------------------------------------------#-------------------------------------------------------------------------------------------------------------------------------#------------------------------------------------------------------------------------------------------------------------------- 
def prepare_banking(df):
    '''drop nulls,
    renamed columns
    drop columns'''
    #Drop amy nulls 
    df= df.dropna()
    #drop columns that i wasnt going to fully uses 
    df=df.drop(columns=['default','month','day_of_week','campaign','poutcome'])
    df = df.rename(columns={'age':'Age','job':'Job','marital':'Marital','education':'Education','housing':'Housing',
                            'loan':'Loan','contact':'Contact','duration':'Duration','pdays':'Pdays'})
    return df



#split_train_test_val-----------------------------------------------------------------------------------------------------------#-------------------------------------------------------------------------------------------------------------------------------#------------------------------------------------------------------------------------------------------------------------------- 

def my_train_test_split(df, target):
    '''
    Takes in a dataframe and target (as a string). Returns train, validate, and test subset 
    dataframes with the .2/.8 and .25/.75 splits to create a final .2/.2/.6 split between datasets
    '''
    train, test = train_test_split(df, test_size = .2, random_state=123, stratify=df[target])
    train, validate = train_test_split(train, test_size = .25, random_state=123, stratify=train[target])
    print(f'Train=',train.shape) 
    print(f'Validate=',validate.shape) 
    print(f'Test=',test.shape) 
    return train, validate, test






#graph--------------------------------------------------------------------------------------------------------------------------#------------------------------------------------------------------------------------------------------------------------------#------------------------------------------------------------------------------------------------------------------------------- 

def graph_0():
    '''graph explaing y '''
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = 'no', 'yes'
    sizes = [29238, 3712]
    explode = (0, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title('client subscribed')
    plt.show()


def graph_1(df):
    '''graph explaining y on percentages'''
    # used this
    plt.figure(figsize=(18,8))
    ax = sns.histplot(data=df,x='y',hue='y')
    ax.bar_label(ax.containers[0])
    ax.bar_label(ax.containers[1])
    print("Percentage:\n",df["y"].value_counts()/len(df)*100)
    plt.title("client subscribed")
    plt.show()
    
def graph_2(train):
    '''graph comapring age to y ,how age effect y '''
    plt.figure(figsize=(18,8))
    sns.histplot(data=train,x='Age',hue='y',kde=True)
    print("Percentage:\n",train["Age"].value_counts()/len(train)*100)
    plt.title("Does age effect client subscribed")
    plt.show()

    
    
def graph_3(train):
    '''graph comapring marital to y ,how marital change the status of y '''
    plt.figure(figsize=(18,8))
    ax=sns.countplot(data=train,x='Marital',hue='y')
    print("Percentage:\n",train["Marital"].value_counts()/len(train)*100)
    ax.bar_label(ax.containers[0])
    ax.bar_label(ax.containers[1])
    plt.title("Does Marital effect client subscribed")
    plt.show()
    
    
    
def graph_4(train):
    ''' job to y, what type of job does more investing to y'''
    plt.figure(figsize=(18,8))
    sns.histplot(data=train, x="Job", y="y",palette='flare',linewidth=5, edgecolor="black",common_norm=False,common_bins=False,stat='count',multiple="stack",element="poly")
    print("Percentage:\n",train["Job"].value_counts()/len(train)*100)
    plt.title("Does job effect client subscribed")
    plt.show()
    
    
    
def graph_5(train):
    ''' job to y, what type of job does more investing to y'''
    plt.figure(figsize=(18,8))
    ax=sns.histplot(data=train, x="Job", y="Age",discrete=(True, False),cbar=True, cbar_kws=dict(shrink=.75))
    cmap = plt.colormaps[plt.rcParams['image.cmap']].with_extremes(bad='y')
    plt.title("Does age/job effect y")
    plt.show()
    
    
def graph_6(train):
    '''comapring education/age to investment to our comapany '''
    plt.figure(figsize=(20,15))
    sns.swarmplot(data=train,x='Age',y='Education',hue='y',palette='mako')
    plt.title("Does Education effect client subscribed")
    plt.show()


def graph_7(train):
    '''comapring job/age to investment to our comapany '''
    plt.figure(figsize=(20,15))
    sns.swarmplot(data=train,x='Age',y='Job',hue='y',palette='YlOrBr')
    plt.title("Does Education effect client subscribed")
    plt.show()

    
def graph_8(train):
    '''comapring education/age to investment to our comapany '''
    plt.figure(figsize=(20,15))
    ax=sns.violinplot(data=train,x='Age',y='Education',hue='y')
    print("Percentage:\n",train["Education"].value_counts()/len(train)*100)
    plt.title("Does education effect client subscribed")
    plt.show()

    
def graph_9(train):
    '''comapring Pdays/age to investment to our comapany '''
    sns.catplot(data=train,x='Pdays',y='Age',col='y',kind="box",height=15,margin_titles=True)
    plt.show()
    
def graph_10(train):
    '''comapring Pdays/age to investment to our comapany '''
    print("Percentage:\n",train["Pdays"].value_counts()/len(train)*100)
    plt.figure(figsize=(20,10))
    g=sns.swarmplot(data=train,x='Pdays',y='Age',hue='y')
    g.set(xlim=(0, 30))
    plt.show()
    
def graph_11(train):
    '''comapring Marital/age to investment to our comapany '''
    sns.displot(data=train, x="Age", hue="Marital", col="y",height=10,
    aspect=.8)
    plt.show()
#stats--------------------------------------------------------------------------------------------------------------------------#-------------------------------------------------------------------------------------------------------------------------------#------------------------------------------------------------------------------------------------------------------------------- 


def test0(train):
    '''#T-Test: Two-sample, Two-tail'''
    before = train.Age<=45
    after = train.Age>=45
    α = 0.05

    s, pval = stats.levene(before, after)
    t, p = stats.ttest_ind(before, after, equal_var=(pval > α))

    if p < α:
        print('Reject the Null Hypothesis. different')
    else:
        print('Fail to reject the Null Hypothesis')
        
        
#stats test
def test1(train):
    ''' 
    This function takes in the train dataset and outputs the Chi-Square results for hypothesis 2b
    in the zillow regression project addressing the relationship between bedroom and bathroom counts.
    '''
    # Set alpha
    α = 0.05

    # Create observed data
    observed = pd.crosstab(train.Marital, train.y)

    # Run chi-square test
    chi2,pval,degf,expected = stats.chi2_contingency(observed)

    # Evaluate results by comparing the p-value with alpha
    if pval < α:
        print('''Reject the Null Hypothesis.
    Findings suggest there is an association between Marital and y.''')
    else:
        print('''Fail to reject the Null Hypothesis.
    Findings suggest there is not an association between Marital and y.''')
        
def test2(train):
    ''' 
    This function takes in the train dataset and outputs the Chi-Square results for hypothesis 2b
    in the zillow regression project addressing the relationship between bedroom and bathroom counts.
    '''
    # Set alpha
    α = 0.05

    # Create observed data
    observed = pd.crosstab(train.Job, train.y)

    # Run chi-square test
    chi2,pval,degf,expected = stats.chi2_contingency(observed)

    # Evaluate results by comparing the p-value with alpha
    if pval < α:
        print('''Reject the Null Hypothesis.
    Findings suggest there is an association between job and y.''')
    else:
        print('''Fail to reject the Null Hypothesis.
    Findings suggest there is not an association between job and y.''')


def test3(train):
    ''' 
    This function takes in the train dataset and outputs the Chi-Square results for hypothesis 2b
    in the zillow regression project addressing the relationship between bedroom and bathroom counts.
    '''
    # Set alpha
    α = 0.05

    # Create observed data
    observed = pd.crosstab(train.Education, train.y)

    # Run chi-square test
    chi2,pval,degf,expected = stats.chi2_contingency(observed)

    # Evaluate results by comparing the p-value with alpha
    if pval < α:
        print('''Reject the Null Hypothesis.
    Findings suggest there is an association between Education and y.''')
    else:
        print('''Fail to reject the Null Hypothesis.
    Findings suggest there is not an association between Education and y.''')


def test4(train):
    '''#T-Test: Two-sample, Two-tail'''
    before = train.Pdays<=27
    after = train.Pdays>=27
    α = 0.05

    s, pval = stats.levene(before, after)
    t, p = stats.ttest_ind(before, after, equal_var=(pval > α))

    if p < α:
        print('Reject the Null Hypothesis. different')
    else:
        print('Fail to reject the Null Hypothesis')

#modeling-----------------------------------------------------------------------------------------------------------------------#-------------------------------------------------------------------------------------------------------------------------------#------------------------------------------------------------------------------------------------------------------------------- 
def getting_ready(train,validate,test):
    ''' drop columns and spliting into x_train,y_train,x_val,y_val,x_test,y_test'''
    X_train = train.drop(columns=['Job','Marital','Education','Housing','Loan','Contact','y'])
    y_train = train.y

    X_validate = validate.drop(columns=['Job','Marital','Education','Housing','Loan','Contact','y'])
    y_validate = validate.y

    X_test = test.drop(columns=['Job','Marital','Education','Housing','Loan','Contact','y'])
    y_test = test.y
    
    return (X_train,y_train,X_validate,y_validate,X_test,y_test)

def score_models(X_train, y_train, X_validate, y_validate):
    '''
    Score multiple models on train and validate datasets.
    Print classification reports to decide on a model to test.
    Return each trained model, so I can choose one to test.
    models = dt_model1, rf_model, knn1_model.
    '''
    dt_model1 = DecisionTreeClassifier(max_depth = 5, random_state = 123)
    rf_model = RandomForestClassifier(min_samples_leaf = 1, max_depth = 10)
    knn1_model = KNeighborsClassifier()
    models = [dt_model1, rf_model, knn1_model]
    for model in models:
        model.fit(X_train, y_train)
        actual_train = y_train
        predicted_train = model.predict(X_train)
        actual_validate = y_validate
        predicted_validate = model.predict(X_validate)
        print(model)
        print('')
        print('train score: ')
        print(classification_report(actual_train, predicted_train))
        print('validate score: ')
        print(classification_report(actual_validate, predicted_validate))
        print('________________________')
        print('')
    return dt_model1, rf_model, knn1_model


def baseline1(train):
    # determine the percentage of customers that churn/do not churn
    baseline = train.y.value_counts().nlargest(1) / train.shape[0]
    print(f'My baseline accuracy is {round(baseline.values[0] * 100,2)}%.')
#best model---------------------------------------------------------------------------------------------------------------------#------------------------------------------------------------------------------------------------------------------------------#-------------------------------------------------------------------------------------------------------------------------------


def best_model(X_test,y_test):
    '''acquiring the best model aka decisiontree testing'''
    #best model we created 
    dt_model1 = DecisionTreeClassifier(max_depth = 5, random_state = 123)
    dt_model1.fit(X_test, y_test)
    actual_test = y_test
    predicted_test = dt_model1.predict(X_test)
    print(classification_report(actual_test, predicted_test))