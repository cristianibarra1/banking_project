![alt text](https://www.springboard.com/blog/wp-content/uploads/2022/05/data-science-life-cycle.png)

# ![alt text](https://imageio.forbes.com/specials-images/imageserve/5f69d5c34f82e0fd92afc183/Online-Banking/960x0.jpg?format=jpg&width=960)

# <hr style="border-bottom: 10px groove black; margin-top: 1px; margin-bottom: 1px"></hr>

# Banking_ Project
by cristian ibarra 

#  <hr style="border-bottom: 10px groove black; margin-top: 1px; margin-bottom: 1px"></hr>

# Data Set Information

The data is related to direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be subscribed ('yes') or not ('no') subscribed.

Goal:- The classification goal is to predict if the client will subscribe (yes/no) a term deposit (variable y).

The dataset contains train,val and test data. Features of train data are listed below.


# Project Objectives

Document code, process (data acquistion, preparation, exploratory data analysis and statistical testing, modeling, and model evaluation), findings, and key takeaways in a Jupyter Notebook Final Report.

Create modules (wrangle.py) that make your process repeateable and your report (notebook) easier to read and follow.

Ask exploratory questions of your data that will help you understand more about the attributes and drivers of home value. Answer questions through charts and statistical tests.

Construct an classification model that predict wether a client would  client subscribed or not.

Make recommendations to a data science team about how to improve predictions.

Be prepared to answer panel questions about your code, process, findings and key takeaways, and model.

# Project description
1)Why this project-
This project would help determind what're the main key aspect of client subcribetions and why our clientle is leaving.

2)Why is this important-
So we could predict our customer and increase our client subcribetions.

3)How does this help you- 
This would help all of us on understanding how and why our clients our staying or leaving.

# <hr style="border-bottom: 10px groove black; margin-top: 1px; margin-bottom: 1px"></hr>


# Business Goals:

`My goal is to find key driver for  client subscribedion`

There has been a revenue decline in a Portuguese Bank and they would like to know what actions to take. After investigation, they found that the root cause was that their customers are not investing enough for long term deposits. So the bank would like to identify existing customers that have higher chance to subscribe for a long term deposit and focus marketing efforts on such customers.

Construct an classification model that predict wether a client would  client subscribed or not.

Deliver a report that the data science team can read through and replicate, understand what steps were taken, why and what the outcome was.

Make recommendations on what works or doesn't work in prediction  client subscribedion.

# Project steps:

Step 1: Understanding the Problem.

Step 2: Data Extraction.

Step 3: Data Cleaning.

Step 4: Exploratory Data Analysis.

Step 5: Feature Selection.

Step 6: Testing the Models.

Step 7: Deploying the Model.
# Project Planning:

•Create README.md with data dictionary, project and business goals, come up with initial hypotheses.

•Acquire data from the Codeup Database and create a function to automate this process. Save the function in an acquire.py file to import into the Final Report Notebook.

•Clean and prepare data for the first iteration through the pipeline, MVP preparation. Create a function to automate the process, store the function in a prepare.py module, and prepare data in Final Report Notebook by importing and using the funtion.

•Clearly define four hypotheses, set an alpha, run the statistical tests needed, reject or fail to reject the Null Hypothesis, and document findings and takeaways.

•Establish a baseline accuracy and document well.

•Evaluate models on train and validate datasets.

•Choose the model with that performs the best and evaluate that single model on the test dataset.

•Document conclusions, takeaways, and next steps in the Final Report Notebook.

# Hypotheses:

`Hypothesis 1 - Does the age effect y`:

Ho-Does 45 and below effect y more then 45 and above?

Ha-45 and below doesnt effect y more then 45 and above?

`Hypothesis 2 - Does Marital status effect client subscribed `:

Ho-Does marital status effect client subcribers

Ha-Marital status doesnt have a impact on client subcribers


`Hypothesis 3 - Does the job effect client subscribed`:

Ho-Does the type of job effect if client would subcribed

Ha-The type of job does't effect if client would subcribed 

`Hypothesis 4 -Does the Education have a effect on client subscribed `:

Ho-Does education have a connection with client subscribed 

Ha-Education doesnt have a connection with client subscribed 

`Hypothesis 5 - Pdays with client subscribed`:

Ho-Does calling more effect client subscribed ?

Ha-Calling more doesnt have a impact on client subscribed 




# Questions:

• What effect client subcribetion more then anything?

• Does marital effect client subcribetion??

• Can we increase our client subcribetion??

# <hr style="border-bottom: 10px groove black; margin-top: 1px; margin-bottom: 1px"></hr>

# <hr style="border-bottom: 10px groove black; margin-top: 1px; margin-bottom: 1px"></hr>

# Data Dictionary/Findings:

# Features
|feature|Feature_Type|Description|
| -------- |-------- | -------- | 
age|numeric|age of a person
job|Categorical,nominal|type of job ('admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
marital|categorical,nominal|marital status ('divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
education|categorical,nominal|('basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
default|categorical,nominal|has credit in default? ('no','yes','unknown')
housing|categorical,nominal|has housing loan? ('no','yes','unknown')
loan|categorical,nominal|has personal loan? ('no','yes','unknown')
contact|categorical,nominal|contact communication type ('cellular','telephone')
month|categorical,ordinal|last contact month of year ('jan', 'feb', 'mar', ..., 'nov', 'dec')
day_of_week|categorical,ordinal|last contact day of the week ('mon','tue','wed','thu','fri')
duration|numeric|last contact duration, in seconds . Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no')
campaign|numeric|number of contacts performed during this campaign and for this client (includes last contact)
pdays|numeric|number of days that passed by after the client was last contacted from a previous campaign (999 means client was not previously contacted)
previous|numeric|number of contacts performed before this campaign and for this client
poutcome|categorical,nominal|outcome of the previous marketing campaign ('failure','nonexistent','success')







# Target variable:

|Feature	|Feature_Type	|Description|
| -------- |-------- | -------- | 
y	|binary|	has the client subscribed a term deposit? ('yes','no')

# <hr style="border-bottom: 10px groove black; margin-top: 1px; margin-bottom: 1px"></hr>

# Modeling:

|Split|Model| precision|recall|f1-score|
| ----- | ----- | ----- | ----- |----- |
|train|DecisionTreeClassifier(max_depth=5, random_state=123)|0.93| 0.97|0.91|
|validate|DecisionTreeClassifier(max_depth=5, random_state=123)|0.93|0.97|0.90|
|train|KNeighborsClassifier()| 0.94|0.98|0.92|
|validate|KNeighborsClassifier()|  0.92|0.97|0.90|
|train|RandomForestClassifier(max_depth=10))|0.94| 0.99 | 0.93|
|validate|RandomForestClassifier(max_depth=10))| 0.92|0.97 |0.90|

---
# VALIDATE:
|Model| precision|recall|f1-score|
| ----- | ----- | ----- |----- |
|DecisionTreeClassifier(max_depth=5, random_state=123)|0.93|0.97|0.90|
|KNeighborsClassifier()|  0.92|0.97|0.90
|RandomForestClassifier(max_depth=10))| 0.92|0.97 |0.90 






# TRAIN:
|Model| precision|recall|f1-score|
| ----- | ----- | ----- |----- |
|DecisionTreeClassifier(max_depth=5, random_state=123)|0.93| 0.97|0.91|
|KNeighborsClassifier()| 0.94|0.98|0.92
|RandomForestClassifier(max_depth=10))|0.94| 0.99 | 0.93 



# Test: 
|Model| precision|recall|f1-score|
| ----- | ----- | ----- |----- |
|DecisionTreeClassifier(max_depth=5, random_state=123)|0.94|0.96|0.91|

<hr style="border-bottom: 10px groove black; margin-top: 1px; margin-bottom: 1px"></hr>

# Conclusion/Recommnedations/Next Steps:
`Conclusion:`

• We could conclude that all the feature we pulled had a impact on client subcribetion,some of them impact more then others...Age was the biggest impact then job,education.

• In conclusion client are not subcribing because we dont know how to reach our target goal... more uneducated client are subcribing the educated ones.

• We could conclude that the DecisionTreeClassifier perform the best with a f1-score of 90% performing 7 percent better then the baseline.


`Recommendations:`

• I would recommend that we future our research on age and find out why middle age adult arent subcribing to us,so we could reach another target audionce .

• I would also reccomend that we teach and futher our information on our client subcribion on how it could help the clients increase them self too. 

• I would also recommend that we updated them about any changes or keep them in the loop.

`Next Steps:`

• My next set would be to find out how much did our y change over the period of 5 years to acquire the right information were doing...for example (Did we contact them more in the past???,Did we give out more information???Did marital status change???. This data has so much to look into but i would just be digging myself into a rabbit hole.

• I would love to check if my recommendation did a impact on our future data

• I would love to uses data the next time around so i could find out what impact then had.

# <hr style="border-bottom: 10px groove black; margin-top: 1px; margin-bottom: 1px"></hr>


# Deliverables/Steps to reproduce finalnotebook:
banking data set was collected from kaggle(https://www.kaggle.com/code/rashmiranu/banking-dataset-eda-and-binary-classification/notebook).Using this data to find what effect our clients

1)Download the following files 

• Wrangle.py

• finalnotebook.pynd

2) After downloading files make sure all files are in the same folder or location 

3) Onces step two and step one are done you would be able to run finalnotebook without errors and on your own 

