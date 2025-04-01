import pandas as pd
import math
import random
import statistics
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import openpyxl
from openpyxl.styles import PatternFill
import squarify
import stemgraphic
import plotly.graph_objects as go
import plotly.express as px
import scipy.stats as stats
import mlxtend
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.frequent_patterns import fpgrowth

data = pd.read_csv(r"K:\Users\keerthi\intership\drugs_side_effects_drugs_com.csv")
data
print(f"Num of attributes: {data.shape[1]}")
print(f"Num of rows of data: {data.shape[0]}")
data = data.drop(['csa', 'drug_link', 'medical_condition_url'], axis=1)
data
columns_to_be_modified = ['pregnancy_category', 'alcohol']

# Every other attributes with null values is indication that the data is not available

for i in data.index:
    for j in columns_to_be_modified:
        check_ = data[j].isnull()
        if check_[i] == True:
            data.loc[i, j] = 0
            
data

# Remove any empty values
data1 = data
data_1 = data1.dropna(axis=0)
data_1

unique_medical_conditions = list(data.medical_condition.unique())
unique_drugs = list(data.drug_name.unique())


# Tokenization
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('stopwords')
import string

def tokenize(document):
    words = []
    document = str(document)
    lower_case = document.lower()
    tokenized = nltk.tokenize.word_tokenize(document)

    # Remove punctuations and english stopwords
    for word in tokenized:
        if(word not in string.punctuation and word not in nltk.corpus.stopwords.words("english")):
            words.append(word)
    return words

side_effects = []
for i in data.index:
    words = tokenize(data['side_effects'][i])
    words = list(set(words))
#     for word in words:
#         side_effects.append(word)
    side_effects.append(words)


    # Related Drugs
related_drugs = []
for i in data.index:
    strings = []
    string = ""
    for aa in range(len(str(data["related_drugs"][i]))):
        if str(data["related_drugs"][i])[aa] == ":":
            strings.append(string)
            string = ""
            break
        else:
            string += str(data["related_drugs"][i])[aa]
    string = ""
    for j in range(len(str(data["related_drugs"][i]))):
        if str(data["related_drugs"][i])[j] == "|":
            for k in range(j+2, len(str(data["related_drugs"][i]))):
                if str(data["related_drugs"][i][k]) == ":":
                    strings.append(string)
                    string = ""
                    break
                else:
                    string += str(data["related_drugs"][i][k])
            j = k
    related_drugs.append(strings)


    # Tokenization
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('stopwords')
import string

def tokenize(document):
    words = []
    document = str(document)
    lower_case = document.lower()
    tokenized = nltk.tokenize.word_tokenize(document)

    # Remove punctuations and english stopwords
    for word in tokenized:
        if(word not in string.punctuation and word not in nltk.corpus.stopwords.words("english")):
            words.append(word)
    return words

# Medical Condition Description
medical_condition_descriptions = []
for i in range(len(unique_medical_conditions)):
    for j in data.index:
        if str(data["medical_condition"][j]) == unique_medical_conditions[i]:
            medical_condition_descriptions.append(data["medical_condition_description"][j])
            break
            
tokenized_medical_condition_descriptions = []

for i in medical_condition_descriptions:
    words = tokenize(i)
    words = list(set(words))
    tokenized_medical_condition_descriptions.append(words)


    def load_data():
    data = pd.read_csv(r"C:\Users\dhyut\SEM8\ASBD LAB\CourseProject\drugs_side_effects_drugs_com.csv")
    data = data.drop(['csa', 'drug_link', 'medical_condition_url'], axis=1)
    columns_to_be_modified = ['pregnancy_category', 'alcohol']
    # Every other attributes with null values is indication that the data is not available
    for i in data.index:
        for j in columns_to_be_modified:
            check_ = data[j].isnull()
            if check_[i] == True:
                data.loc[i, j] = 0
    return data

data = load_data()
for i in side_effects:
    i = set(i)
    i = list(i)

sample = 0
print("SAMPLE TOKENIZED DATA\n")
print(f"Drug_name: {data['drug_name'][sample]}\n\nMedical Condition: {data['medical_condition'][sample]}\n\nSide Effects: {side_effects[sample]}\n\nRelated Drugs: {related_drugs[sample]}\n\n Medical Condition Descriptions: {tokenized_medical_condition_descriptions[sample]}")
print("After data preprocessing, data is as follows:")
data

print(f"Dimension of the dataset: {data.shape}\n")
cnt = 0
for i in data.columns:
    print(f"Mode in column {data.columns[cnt]}: {statistics.mode(data[data.columns[cnt]])}")
    cnt += 1
print("\n")

df =data
df.info()
# calculate duplicates
dups = df.duplicated()
# report if there are any duplicates
print(df.any())
# list all duplicate rows

print("Duplicate Rows",df[dups])
#Cardinality 
df.nunique() # To determine the maximum and minimum number of variations in each column of the dataset
#Lets now check for null fields
import seaborn as sns
plt.figure(figsize=(10,12))
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
df.isnull().sum()
print(f"Data types of each column:\n{data.dtypes}\n")

# Group data accordingly
unique_medical_conditions = list(data.medical_condition.unique())
unique_drugs = list(data.drug_name.unique())

print(f"List of unique Medical Conditions being discussed are:\n{unique_medical_conditions}\n\n")
print(f"List of unique drugs being discussed are:\n{unique_drugs}\n\n")
print(f"Number of unique drugs: {len(unique_drugs)}")

# Drugs being used for multiple medical conditions
drugs_multiple_condns = []
drugs_so_far = []

for i in data.index:
    if data["drug_name"][i] in drugs_so_far and data["drug_name"][i] not in drugs_multiple_condns:
        drugs_multiple_condns.append(data["drug_name"][i])
    if data["drug_name"][i] not in drugs_so_far:
        drugs_so_far.append(data["drug_name"][i])

print("The below is the list of only drugs that have multi-purpose:")
for i in drugs_multiple_condns:
    print(i)

    available_drugs_for_condns = [0]*len(unique_medical_conditions)

for i in data.index:
    medical_condition = str(data['medical_condition'][i])
    indexx = unique_medical_conditions.index(medical_condition)
    available_drugs_for_condns[indexx] += 1

#Bar Graph
fig = plt.figure(figsize = (10, 5))
# creating the bar plot
plt.bar(unique_medical_conditions, available_drugs_for_condns, width = 0.5)
plt.xlabel("Medical Conditions")
plt.ylabel(f"Number of drugs available")
plt.title(f"Medical Conditions vs Number of Drugs available")
plt.xticks(rotation=90)
plt.show()
print(f"Total data rows: {data.shape[0]}")

# Pareto Chart
available_drugs_for_condns, unique_medical_conditions = zip(*sorted(zip(available_drugs_for_condns, unique_medical_conditions)))

#Bar Graph
fig = plt.figure(figsize = (10, 5))
# creating the bar plot
plt.bar(unique_medical_conditions, available_drugs_for_condns, width = 0.5)
plt.xlabel("States")
plt.ylabel(f"Number of drugs available")
plt.title(f"Medical Conditions vs Number of Drugs available - Sorted")
plt.xticks(rotation=90)


unique_rx_otc = ['nil', 'OTC', 'Rx', 'Rx/OTC']

cnts_rtx_otc = [0]*len(unique_rx_otc)
for i in data.index:
    if data["rx_otc"][i] == None:
        cnts_rtx_otc[0] += 1
    elif data["rx_otc"][i] == 'OTC':
        cnts_rtx_otc[1] += 1
    elif data["rx_otc"][i] == 'Rx':
        cnts_rtx_otc[2] += 1
    elif data["rx_otc"][i] == 'Rx/OTC':
        cnts_rtx_otc[3] += 1
        
# Pareto Chart
# cnts, unique_rx_otc = zip(*sorted(zip(cnts, unique_rx_otc)))

#Bar Graph
fig = plt.figure(figsize = (10, 5))
# creating the bar plot
plt.bar(unique_rx_otc, cnts_rtx_otc, width = 0.5)
plt.xlabel("Rx/OTC values of drugs")
plt.ylabel(f"Counts")
plt.title(f"Histogram of Rx OTC counts")
plt.xticks(rotation=90)
plt.show()
plt.show()

pregnancy_categories = ['A', 'B', 'C', 'D', 'X', 'N']

cnts_preg_cat = [0]*len(pregnancy_categories)
for i in data.index:
    if data["pregnancy_category"][i] == 'A':
        cnts_preg_cat[0] += 1
    elif data["pregnancy_category"][i] == 'B':
        cnts_preg_cat[1] += 1
    elif data["pregnancy_category"][i] == 'C':
        cnts_preg_cat[2] += 1
    elif data["pregnancy_category"][i] == 'D':
        cnts_preg_cat[3] += 1
    elif data["pregnancy_category"][i] == 'X':
        cnts_preg_cat[4] += 1
    elif data["pregnancy_category"][i] == 'N':
        cnts_preg_cat[5] += 1
    
        
# Pareto Chart
#Bar Graph
fig = plt.figure(figsize = (10, 5))
# creating the bar plot
plt.bar(pregnancy_categories, cnts_preg_cat, width = 0.5)
plt.xlabel("Pregnancy Category")
plt.ylabel(f"Counts")
plt.title(f"Histogram of Pregnancy Categories")
plt.xticks(rotation=0)
plt.show()


drug_names = []
no_of_ratings = []

num = 50

cnt = 0
for i in data.index:
    if cnt != num:
        drug_names.append(data["drug_name"][i])
        if not np.isnan(data["no_of_reviews"][i]):
            no_of_ratings.append(int(data["no_of_reviews"][i]))
        else:
            no_of_ratings.append(0)
    else:
        break
    cnt += 1

#Bar Graph
fig = plt.figure(figsize = (10, 5))
no_of_ratings, drug_names = zip(*sorted(zip(no_of_ratings, drug_names)))
# creating the bar plot
# num = 10
plt.bar(drug_names, no_of_ratings, width = 0.5)
plt.xlabel("Drug Name")
plt.ylabel(f"Number of RegisteredUsers")
plt.title(f"Histogram of Reviews - first {num} drugs - pareto (sorted)")
plt.xticks(rotation=90)
plt.show()

# Ratings
num = 50
drug_names = data["drug_name"][:num]
ratings = data["rating"][:num]
ratings, drug_names = zip(*sorted(zip(ratings, drug_names)))        
# Pareto Chart
#Bar Graph
fig = plt.figure(figsize = (10, 5))
# creating the bar plot
plt.bar(drug_names, ratings, width = 0.5)
plt.xlabel("Drug Name")
plt.ylabel(f"Ratings")
plt.title(f"Drug Name vs Ratings - Top {num}")
plt.xticks(rotation=90)
plt.show()

# Examining the variable's goal rating
sns.set_context('poster', font_scale=0.8)
plt.figure(figsize=(35,5))
sns.countplot(data=data,x = 'rating',palette='plasma')

# Rx/OTC distributions
y = np.array(cnts_rtx_otc)
mylabels = unique_rx_otc
myexplode = [0.2]*len(unique_rx_otc)
plt.title("\nPie Chart - Rx/OTC Distribution of drugs")
plt.pie(y, labels = mylabels, explode = myexplode, autopct='%.0f%%')
plt.show()

# Figure size 
sns.set_context('poster', font_scale=0.5)
plt.figure(figsize=(10,10))
# Pie plot
data['medical_condition'].value_counts().plot.pie(autopct='%1.1f%%', textprops={'fontsize':12}).set_title("Target distribution on medical_condition ")
data['medical_condition'].value_counts()

# Alcohol
cnts = 0

for i in data.index:
    if data["alcohol"][i] == 'X':
        cnts += 1
cnts_alcohol = [cnts, data.shape[0] - cnts]
mylabels = ["Reacts with alcohol", "No reaction with alcohol"]

# Alcohol
y = np.array(cnts_alcohol)
# mylabels = unique_rx_otc
myexplode = [0.01]*2
plt.title("Pie Chart - Distribution of drugs based on their reaction with alcohol")
plt.pie(y, labels = mylabels, explode = myexplode, autopct='%.0f%%')
plt.show()

# Alcohol
cnts = 0

for i in data.index:
    if data["alcohol"][i] == 'X':
        cnts += 1
cnts_alcohol = [cnts, data.shape[0] - cnts]
mylabels = ["Reacts with alcohol", "No reaction with alcohol"]

# Alcohol
y = np.array(cnts_alcohol)
# mylabels = unique_rx_otc
myexplode = [0.01]*2
plt.title("Pie Chart - Distribution of drugs based on their reaction with alcohol")
plt.pie(y, labels = mylabels, explode = myexplode, autopct='%.0f%%')
plt.show()

# Medical Conditions vs Ratings

avg_ratings = [0]*len(unique_medical_conditions)
avg_denominator = [0]*len(unique_medical_conditions)
avg_activities = [0]*len(unique_medical_conditions)
for i in range(len(unique_medical_conditions)):
    for j in data.index:
        if str(data["medical_condition"][j]) == unique_medical_conditions[i] and not np.isnan(data["rating"][j]):
            avg_ratings[i] += float(data["rating"][j])
            avg_denominator[i] += 1
            avg_activities[i] += float(data["activity"][j][:-1])

for i in range(len(unique_medical_conditions)):
    if avg_denominator[i] != 0:
        avg_ratings[i] = avg_ratings[i]/avg_denominator[i]
        avg_activities[i] = (avg_activities[i]/avg_denominator[i])/10

# for i in range(len(unique_medical_conditions)):
#     if avg_ratings[i]:
#         avg_ratings[i] = 0


plt.plot(unique_medical_conditions, np.array(avg_ratings))
plt.plot(unique_medical_conditions, np.array(avg_activities))
plt.xlabel(f"Problem Name")
plt.ylabel(f"Ratings of available medicines")
plt.title(f"Activities and Ratings of drugs - Line Plot")
plt.legend(["Avg_Ratings", "Avg_Activities"])
plt.xticks(rotation=90)
plt.show()

plt.scatter(unique_medical_conditions, np.array(avg_ratings))
plt.scatter(unique_medical_conditions, np.array(avg_activities))
plt.xlabel(f"Problem Name")
plt.ylabel(f"Ratings of available medicines")
plt.title(f"Activities and Ratings of drugs - Scatter Plot")
plt.legend(["Avg_Ratings", "Avg_Activities"])
plt.xticks(rotation=90)


num = 4
diseases = unique_medical_conditions[:num]
property_chosen = [data["rating"], data["activity"]]

properties = []

for i in diseases:
    dataa = data.loc[data['medical_condition'] == i]
    rating = list(dataa["rating"])
    activity = []
    for j in dataa.index:
        activity.append(int(dataa["activity"][j][:-1]))
    plt.scatter(rating, activity)
    plt.xlabel(f"Ratings")
    plt.ylabel(f"Acticity")
    plt.title(f"Activities and Ratings of drugs - Scatter Plot")
    plt.legend(diseases)
    plt.xticks(rotation=90)
plt.show()

# available_drugs_for_condns
plt.boxplot(available_drugs_for_condns)
plt.title(f"Box Plot of Number of available drugs for each disease")
plt.show()

plt.boxplot(activities)
plt.title(f"Box Plot of activites for each disease")
plt.show()

plt.boxplot(cnts_preg_cat)
plt.title(f"Box-plot to see the pregnancy categories of drugs")
plt.show()

import plotly.express as ex
ex.box(x = 'rx_otc', y = 'medical_condition',data_frame = data, template='seaborn', notched=True, width=800, height=500)
ex.box(x = 'medical_condition', y = 'drug_classes',data_frame = data, template='seaborn', notched=True, width=1200, height=600)

sns.violinplot(available_drugs_for_condns)
plt.xlabel(f"Drugs distribution")
plt.ylabel(f"Number of available drugs")
plt.title(f"Violin Plot of Drugs Distribuion over each disease")
plt.xticks(rotation=90)
plt.show()

sns.violinplot(activities)
plt.xlabel(f"Drugs distribution")
plt.ylabel(f"Activity %")
plt.title(f"Violin Plot of Drugs Distribuion over each disease - Activity %")
plt.xticks(rotation=90)
plt.show()


data = pd.read_csv("K:\Users\keerthi\intership\drugs_side_effects_drugs_com.csv")
num = 50
activity = []
for i in data.index:
    activity.append(int(data["activity"][i][:-1]))
# print(f"Number of Reviews of the first {num} drugs")
stemgraphic.stem_graphic(activity[:num] , scale = 10)

categories = ["activity", "rating", "no_of_reviews"]

print("Radar Chart")
fig = go.Figure()
plt.Figure(figsize =(5, 3))

fig.add_trace(go.Scatterpolar(r=data.loc[data["medical_condition"] == 'Acne'], theta=categories, fill='toself', name='acne'))
fig.add_trace(go.Scatterpolar(r=data.loc[data["medical_condition"] == 'Anxiety'], theta=categories, fill='toself', name='anxiety'))
fig.add_trace(go.Scatterpolar(r=data.loc[data["medical_condition"] == 'Asthma'], theta=categories, fill='toself', name='asthma'))
fig.add_trace(go.Scatterpolar(r=data.loc[data["medical_condition"] == 'Weight Loss'], theta=categories, fill='toself', name='weight loss'))
fig.update_layout(autosize=False, width=400, height=400)

# Data as a dictionary

arr = []
# for i in data.columns[3:]:
#     arr.append(math.ceil(statistics.mean(data1[i])))
#available_drugs_for_condns, unique_medical_conditions = zip(*sorted(zip(available_drugs_for_condns, unique_medical_conditions)))
data = dict(time=available_drugs_for_condns,
            steps=unique_medical_conditions)

fig = px.funnel(data, x='time', y='steps')
print("Funnel Chart for Diseases and Available drugs")
fig.show()

data = load_data()
activity = []
rating = []
for i in data.index:
    activity.append(int(data["activity"][i][:-1]))
    if not np.isnan(data["rating"][i]):
            rating.append(int(data["rating"][i]))
    else:
            rating.append(0)
num = 5
# correlation_matrix = np.corrcoef(activity[:num], rating[:num])
heatmap_sepal = sns.heatmap([rating[:num], activity[:num]], vmin = 0, vmax = 100)
plt.xlabel("Activity")
plt.ylabel("Rating")
plt.show()

data = load_data()
df = data[["medical_condition", "rating", "no_of_reviews"]]
# df.drop(columns=df.columns[0], axis=1,  inplace=True)
sns.pairplot(df, kind="scatter", hue="medical_condition", palette="Set1")
plt.title("Correlation Matrix")
plt.show()

# Transactions
cnt = 0
for i in data.index:
    related_drugs[cnt].append(data["drug_name"][i])
    cnt += 1
    
print(related_drugs)

te = TransactionEncoder()
te_ary = te.fit_transform(related_drugs)

# Print the one-hot encoded Boolean array
print(te_ary)

df = pd.DataFrame(te_ary, columns=te.columns_)
print(df)



# Use the Apriori algorithm to generate frequent itemsets
frequent_itemsets = apriori(df, min_support=0.01, use_colnames=True)

# Use the Association Rules algorithm to generate rules with a minimum confidence
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

# Print the frequent itemsets and association rules
print("Frequent Itemsets:")
print(frequent_itemsets)

print("\nAssociation Rules:")
print(rules)

# Get the length of the largest frequent itemset
max_len = frequent_itemsets["itemsets"].apply(lambda x: len(x)).max()

# Print only the last level of frequent itemsets
print("Frequent Itemsets (Level {}):".format(max_len))
last_level_itemsets = frequent_itemsets[frequent_itemsets["itemsets"].apply(lambda x: len(x))==max_len]
print(last_level_itemsets)

# Use the Association Rules algorithm to generate rules with a minimum confidence
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

# Print the association rules for the last level frequent itemsets
print("\nAssociation Rules for Frequent Itemsets (Level {}):".format(max_len))
last_level_rules = rules[rules["antecedents"].apply(lambda x: len(x)==max_len)]
print(last_level_rules)

#traversal through database for once took 4 mins
import numpy as np
# Gather All Items of Each Transactions into Numpy Array
from itertools import chain
transaction = list(chain.from_iterable(related_drugs))
# converting to numpy array
transaction = np.array(transaction)
print(transaction)

#  Transform Them a Pandas DataFrame
df = pd.DataFrame(transaction, columns=["items"]) 
# Put 1 to Each Item For Making Countable Table, to be able to perform Group By
df["incident_count"] = 1 
#  Delete NaN Items from Dataset
indexNames = df[df['items'] == "nan" ].index
df.drop(indexNames , inplace=True)
# Making a New Appropriate Pandas DataFrame for Visualizations  
df_table = df.groupby("items").sum().sort_values("incident_count", ascending=False).reset_index()
#  Initial Visualizations
df_table.head(5).style.background_gradient(cmap='Blues')
fig.show()

# to have a same origin
df_table["all"] = "Top 50 items" 
# creating tree map using plotly
fig = px.treemap(df_table.head(50), path=['all', "items"], values='incident_count',
                  color=df_table["incident_count"].head(50), hover_data=['items'],
                  color_continuous_scale='Blues',
                )
# ploting the treemap
fig.show()

te = TransactionEncoder()
te_ary = te.fit_transform(related_drugs)

# Print the one-hot encoded Boolean array
print(te_ary)
dataset = pd.DataFrame(te_ary, columns=te.columns_)

# select top 30 items
first30 = df_table["items"].head(30).values 
# Extract Top 30
dataset = dataset.loc[:,first30] 
# shape of the dataset
dataset.shape

from mlxtend.frequent_patterns.fpgrowth import fpgrowth
#running the fpgrowth algorithm
res=fpgrowth(dataset,min_support=0.05, use_colnames=True)
# printing top 10
res


transactions = related_drugs
print(transactions[:10])

# Generate frequent 1-itemsets
candidate_1_itemsets = []
for transaction in transactions:
    for item in transaction:
        if [item] not in candidate_1_itemsets:
            candidate_1_itemsets.append([item])
print(len(candidate_1_itemsets))                        
print(candidate_1_itemsets)

def prune(itemsets, min_support, transactions):
    frequent_itemsets = []
    for itemset in itemsets:
        support_count = 0
        for transaction in transactions:
            if set(itemset).issubset(set(transaction)):
                support_count += 1
        if support_count >= min_support:
            frequent_itemsets.append(itemset)
    return frequent_itemsets
frequent_itemsets = prune(candidate_1_itemsets, 100 , transactions)
print(len(frequent_itemsets))
print(frequent_itemsets)

def generate_candidates(itemsets, k):
    candidates = []
    for itemset1 in itemsets:
        for itemset2 in itemsets:
            if itemset1 != itemset2 and len(set(itemset1) | set(itemset2)) == k:
                candidate = sorted(list(set(itemset1) | set(itemset2)))
                if candidate not in candidates:
                    candidates.append(candidate)
    return candidates

'''candidate_2_itemsets = generate_candidates(frequent_itemsets,2)
frequent_itemsets = prune(candidate_2_itemsets, 300 , transactions)
print(frequent_itemsets)
print("\n")'''

def flatten(l):
    return set([item for sublist in l for item in sublist])

k=2
while len(frequent_itemsets)>0:
    candidate_itemsets=generate_candidates(frequent_itemsets,k)
    reduced_transactions = []
    for transaction in transactions:
        reduced_transaction = [item for item in transaction if item in set(flatten(frequent_itemsets))]
        if len(reduced_transaction) > 0:
            reduced_transactions.append(reduced_transaction)
    frequent_itemsets = prune(candidate_itemsets, 100 , reduced_transactions)
    print("itemsets of length ",k)
    print(len(frequent_itemsets))
    print(frequent_itemsets)  
    print("\n")
    k+=1

# Import the required libraries
from collections import defaultdict
import itertools

# Define a function to compute the support of each itemset
def compute_support(itemsets, transactions):
    support = defaultdict(int)
    for transaction in transactions:
        for itemset in itemsets:
            if set(itemset).issubset(transaction):
                support[frozenset(itemset)] += 1
    return support

# Define the sample dataset
min_support = 1000

# Find all frequent itemsets and their support values
frequent_itemsets = {}
itemsets = set(item for transaction in transactions for item in transaction)
for k in range(1, len(itemsets)+1):
    candidate_itemsets = [set(itemset) for itemset in itertools.combinations(itemsets, k)]
    support = compute_support(candidate_itemsets, transactions)
    for itemset in support:
        if support[itemset] >= min_support:
            frequent_itemsets[frozenset(itemset)] = support[itemset]

# Initialize the closed frequent itemsets
closed_itemsets = frequent_itemsets.copy()[:10]

# Compute the closure of each frequent itemset
for itemset1 in frequent_itemsets[:10]:
    for itemset2 in frequent_itemsets[:10]:
        if itemset1 != itemset2 and itemset1.issubset(itemset2) and frequent_itemsets[itemset1] == frequent_itemsets[itemset2]:
            closed_itemsets.pop(itemset1, None)
            break

# Print the resulting closed frequent itemsets
print(closed_itemsets)

data = pd.read_csv(r"K:\Users\keerthi\intership\drugs_side_effects_drugs_com.csv")

data = data.replace('nan', np.nan)
data = data.dropna()
data.alcohol.unique()
data.pregnancy_category.unique()

# create a dictionary to map the values
pregnancy_map = {'X': 0, 'B': 1, 'C': 2, 'D': 3, 'N' : 0}

# map the values in the 'pregnancy' column using the dictionary
data['pregnancy_category'] = data['pregnancy_category'].map(pregnancy_map)

data.rx_otc.unique()
# create a dictionary to map the values
rx_map = {'Rx': 0, 'OTC': 1, 'Rx/OTC': 2}

# map the values in the 'pregnancy' column using the dictionary
data['rx_otc'] = data['rx_otc'].map(rx_map)
data.activity.unique()

data['activity'] = data['activity'].str.strip('%')
data.activity.unique()
       
data['activity'] = data['activity'].astype(float)
data.activity.unique()     
data.pregnancy_category.unique()

df=data
# convert each column to a numeric data type
df_numeric = df.apply(pd.to_numeric, errors='coerce')

# drop columns with non-numeric values
df_clean = df_numeric.dropna(axis=1)

# display the updated DataFrame
print(df_clean)

### import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

df=df_clean
# split the data into input and output variables
X = df.drop('rx_otc', axis=1)
y = df['rx_otc']

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# create a decision tree classifier with default parameters
dtc = DecisionTreeClassifier()

# train the model on the training data
dtc.fit(X_train, y_train)

# evaluate the model on the testing data
score = dtc.score(X_test, y_test)

print(f"Decision Tree Accuracy: {score:.3f}")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# perform k-means clustering with k=3
df=df_clean
df=data[['rating', 'activity']]
df = df.dropna()

X = df.values
X = df.values
kmeans = KMeans(n_clusters=3)
y_kmeans = kmeans.fit_predict(X)

# visualize the clusters with a scatter plot
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('K-Means Clustering')
plt.xlabel('rating')
plt.ylabel('activity')
plt.legend()
plt.show()

# Figure size 
sns.set_context('poster', font_scale=0.5)
plt.figure(figsize=(5,5))
# Pie plot
data['pregnancy_category'].value_counts().plot.pie(autopct='%1.1f%%', textprops={'fontsize':12}).set_title("Target distribution on pregnancy_category")
data['pregnancy_category'].value_counts()




print(transactions[:10])
