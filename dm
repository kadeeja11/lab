# Implementation of various data preprocessing, encoding, scaling, and analysis methods without using built-in libraries

import itertools
import random

# Sample dataset for various operations
data = [
    {'Age': 25, 'Salary': 50000, 'City': 'New York', 'Purchased': 'Yes'},
    {'Age': 30, 'Salary': 60000, 'City': 'Los Angeles', 'Purchased': 'No'},
    {'Age': 35, 'Salary': None, 'City': 'Chicago', 'Purchased': 'Yes'},
    {'Age': None, 'Salary': 80000, 'City': 'New York', 'Purchased': 'No'},
    {'Age': 40, 'Salary': 70000, 'City': 'Los Angeles', 'Purchased': 'Yes'}
]

# 1./11. Data Preprocessing - Handling Missing Values
def fill_missing_values(data):
    for column in ['Age', 'Salary']:
        values = [row[column] for row in data if row[column] is not None]
        mean_value = sum(values) / len(values)
        for row in data:
            if row[column] is None:
                row[column] = mean_value
    return data

data = fill_missing_values(data)

# 2. Sampling Methods
def random_sampling(data, sample_size):
    return random.sample(data, sample_size)

sampled_data = random_sampling(data, 3)

# 4. Normalize Data (Max Absolute and Min-Max Scaling)
def min_max_scaling(data, column):
    values = [row[column] for row in data]
    min_val, max_val = min(values), max(values)
    for row in data:
        row[column] = (row[column] - min_val) / (max_val - min_val)
    return data

data = min_max_scaling(data, 'Salary')

# 5. Standardization (Z-Score)
def z_score_standardization(data, column):
    values = [row[column] for row in data]
    mean_val = sum(values) / len(values)
    std_dev = (sum([(x - mean_val) ** 2 for x in values]) / len(values)) ** 0.5
    for row in data:
        row[column] = (row[column] - mean_val) / std_dev
    return data

data = z_score_standardization(data, 'Age')

# 6. Detect and Remove Outliers (Using IQR)
def remove_outliers(data, column):
    values = [row[column] for row in data]
    q1, q3 = sorted(values)[int(len(values) * 0.25)], sorted(values)[int(len(values) * 0.75)]
    iqr = q3 - q1
    lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return [row for row in data if lower_bound <= row[column] <= upper_bound]

data = remove_outliers(data, 'Salary')

# 7. Label Encoding for Categorical Data
def label_encoding(data, column):
    unique_values = {val: idx for idx, val in enumerate(set(row[column] for row in data))}
    for row in data:
        row[column] = unique_values[row[column]]
    return data

data = label_encoding(data, 'City')

# 8. One hot encoding for the categorical attributes 
def one_hot_encoding(data, column):
    unique_values = set(row[column] for row in data)
    for row in data:
        for unique in unique_values:
            row[f"{column}_{unique}"] = 1 if row[column] == unique else 0
        del row[column]
    return data

data = one_hot_encoding(data, 'Purchased')

# 3. SMOTE (Synthetic Minority Over-sampling Technique)
def smote(data, target_column, minority_class, k=2):
    minority_samples = [row for row in data if row[target_column] == minority_class]
    synthetic_samples = []
    
    for _ in range(len(minority_samples)):
        sample1, sample2 = random.sample(minority_samples, 2)
        synthetic_sample = {col: sample1[col] + random.uniform(0, 1) * (sample2[col] - sample1[col]) if isinstance(sample1[col], (int, float)) else sample1[col] for col in sample1}
        synthetic_samples.append(synthetic_sample)
    
    return data + synthetic_samples

data = smote(data, 'Purchased_Yes', 1)

# 9. Target Mean Encoding
def target_mean_encoding(data, column, target):
    category_means = {}
    for row in data:
        category_means.setdefault(row[column], []).append(row[target])
    category_means = {k: sum(v)/len(v) for k, v in category_means.items()}
    for row in data:
        row[column] = category_means[row[column]]
    return data

data = target_mean_encoding(data, 'City', 'Purchased_Yes')

# 10. Frequency Encoding
def frequency_encoding(data, column):
    frequencies = {}
    for row in data:
        frequencies[row[column]] = frequencies.get(row[column], 0) + 1
    for row in data:
        row[column] = frequencies[row[column]]
    return data

data = frequency_encoding(data, 'City')
      
# 12. Binning Methods (Equal-width Binning)
def binning(data, column, bins):
    values = [row[column] for row in data]
    min_val, max_val = min(values), max(values)
    bin_width = (max_val - min_val) / bins
    for row in data:
        row[column] = int((row[column] - min_val) / bin_width)
    return data

data = binning(data, 'Salary', 3)

# 13. Feature Selection Using Chi-Square (Categorical Data)
def chi_square_feature_selection(data, target_column):
    category_counts = {}
    for row in data:
        key = (row[target_column], row['City'])
        category_counts[key] = category_counts.get(key, 0) + 1
    return category_counts

chi_square_result = chi_square_feature_selection(data, 'Purchased_Yes')
          
# 14. APRIORI ALGORITHM
from itertools import combinations

def get_support(itemset, transactions):
    """Calculate the support of an itemset."""
    return sum(1 for transaction in transactions if itemset.issubset(transaction)) / len(transactions)

def generate_candidates(prev_frequent, k):
    """Generate candidate itemsets of length k from the previous frequent itemsets."""
    candidates = set()
    prev_frequent_list = list(prev_frequent)
    
    for i in range(len(prev_frequent_list)):
        for j in range(i + 1, len(prev_frequent_list)):
            union_set = prev_frequent_list[i] | prev_frequent_list[j]
            if len(union_set) == k:
                candidates.add(union_set)
    return candidates

def apriori(transactions, min_support):
    """Apriori algorithm to find frequent itemsets."""
    frequent_itemsets = []
    single_items = {frozenset([item]) for transaction in transactions for item in transaction}
    
    current_frequent = {item for item in single_items if get_support(item, transactions) >= min_support}
    k = 2
    
    while current_frequent:
        frequent_itemsets.extend(current_frequent)
        candidates = generate_candidates(current_frequent, k)
        current_frequent = {itemset for itemset in candidates if get_support(itemset, transactions) >= min_support}
        k += 1
    
    return frequent_itemsets

def generate_association_rules(frequent_itemsets, transactions, min_confidence):
    """Generate strong association rules from frequent itemsets."""
    rules = []
    for itemset in frequent_itemsets:
        if len(itemset) > 1:  # Only generate rules for sets with more than one item
            for i in range(1, len(itemset)):
                for subset in combinations(itemset, i):
                    antecedent = frozenset(subset)
                    consequent = itemset - antecedent
                    support_antecedent = get_support(antecedent, transactions)
                    support_itemset = get_support(itemset, transactions)
                    confidence = support_itemset / support_antecedent if support_antecedent > 0 else 0

                    if confidence >= min_confidence:
                        rules.append((antecedent, consequent, confidence))

    return rules

# Transactions dataset
transactions = [
    {'Milk', 'Bread', 'Eggs'},
    {'Bread', 'Butter', 'Eggs'},
    {'Milk', 'Bread', 'Butter', 'Cheese'},
    {'Milk', 'Butter', 'Cheese', 'Eggs'},
    {'Bread', 'Butter', 'Cheese'},
    {'Milk', 'Bread', 'Eggs'}
]

# Parameters
min_support = 0.3
min_confidence = 0.7  # Example: minimum confidence of 70%

# Run Apriori Algorithm
frequent_itemsets = apriori(transactions, min_support)
print("Frequent Itemsets:", frequent_itemsets)

# Generate and print strong association rules
strong_rules = generate_association_rules(frequent_itemsets, transactions, min_confidence)
print("\nStrong Association Rules:")
for rule in strong_rules:
    print(f"{set(rule[0])} => {set(rule[1])} (Confidence: {rule[2]:.2f})")

#FP TREE
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules

# Define dataset
transactions = [
    {'Milk', 'Bread', 'Eggs'},
    {'Bread', 'Butter', 'Eggs'},
    {'Milk', 'Bread', 'Butter', 'Cheese'},
    {'Milk', 'Butter', 'Cheese', 'Eggs'},
    {'Bread', 'Butter', 'Cheese'},
    {'Milk', 'Bread', 'Eggs'}
]

# Convert transactions into a DataFrame (one-hot encoding format)
unique_items = sorted({item for transaction in transactions for item in transaction})  # Get all unique items
df = pd.DataFrame([{item: (item in transaction) for item in unique_items} for transaction in transactions])

# Apply FP-Growth algorithm
min_support = 0.3
frequent_itemsets = fpgrowth(df, min_support=min_support, use_colnames=True)

# Generate association rules
min_confidence = 0.7
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

# Display results
print("\nFrequent Itemsets:")
print(frequent_itemsets)

print("\nStrong Association Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
