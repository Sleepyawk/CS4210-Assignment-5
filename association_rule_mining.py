#-------------------------------------------------------------------------
# AUTHOR: Jonathan Lu
# FILENAME: association_rule_mining.py
# SPECIFICATION: Read input to find strong rules related to supermarket products
# FOR: CS 4210 - Assignment #5
# TIME SPENT: 2 hr
#-----------------------------------------------------------*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

#Use the command: "pip install mlxtend" on your terminal to install the mlxtend library

#read the dataset using pandas
df = pd.read_csv('C:/Users/Administrator/Desktop/retail_dataset.csv', sep=',')

#find the unique items all over the data an store them in the set below
itemset = set()
for i in range(0, len(df.columns)):
    items = (df[str(i)].unique())
    itemset = itemset.union(set(items))

#remove nan (empty) values by using:
itemset.remove(np.nan)

#To make use of the apriori module given by mlxtend library, we need to convert the dataset accordingly. Apriori module requires a
# dataframe that has either 0 and 1 or True and False as data.
#Example:

#Bread Wine Eggs
#1     0    1
#0     1    1
#1     1    1

#To do that, create a dictionary (labels) for each transaction, store the corresponding values for each item (e.g., {'Bread': 0, 'Milk': 1}) in that transaction,
#and when is completed, append the dictionary to the list encoded_vals below (this is done for each transaction)
#-->add your python code below

items = ['Bread','Wine','Eggs','Meat','Cheese','Pencil','Diaper']
encoded_vals = []
for index, row in df.iterrows():
    transactions = row.to_numpy()
    labels = {}
    for item in items:
        if item in transactions:
            labels[item] = 1
        else:
            labels[item] = 0
    encoded_vals.append(labels)

#adding the populated list with multiple dictionaries to a data frame
ohe_df = pd.DataFrame(encoded_vals)

#calling the apriori algorithm informing some parameters
freq_items = apriori(ohe_df, min_support=0.2, use_colnames=True, verbose=1)
rules = association_rules(freq_items, metric="confidence", min_threshold=0.6)

#iterate the rules data frame and print the apriori algorithm results by using the following format:

#Meat, Cheese -> Eggs
#Support: 0.21587301587301588
#Confidence: 0.6666666666666666
#Prior: 0.4380952380952381
#Gain in Confidence: 52.17391304347825
#-->add your python code below

def getSupportCount(df, consequent):
    supportCount = 0
    for index, row in df.iterrows():
        transaction = row.to_numpy()
        if consequent in transaction:
            supportCount += 1
    return supportCount

for index, row in rules.iterrows():
    print(list(row['antecedents']),'->',list(row['consequents']))
    print('Support:', row['support'])
    print('Confidence:', row['confidence'])

#To calculate the prior and gain in confidence, find in how many transactions the consequent of the rule appears (the supporCount below). Then,
#use the gain formula provided right after.
#prior = suportCount/len(encoded_vals) -> encoded_vals is the number of transactions
#print("Gain in Confidence: " + str(100*(rule_confidence-prior)/prior))
#-->add your python code below
    rule_confidence = row['confidence']
    consequent = list(row['consequents'])
    supportCount = getSupportCount(df, consequent)
    prior = supportCount / len(encoded_vals)
    print('Prior:', prior)
    print("Gain in Confidence: " + str(100*(rule_confidence-prior)/prior))
    print('\n')

#Finally, plot support x confidence
plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.title('Support vs Confidence')
plt.show()