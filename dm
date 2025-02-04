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

def max_absolute_scaling(data, column):
    values = [abs(row[column]) for row in data if row[column] is not None]  # Handle None values
    max_abs_val = max(values) if values else 1  # Avoid division by zero
    
    for row in data:
        if row[column] is not None:
            row[column] = row[column] / max_abs_val  # Scale each value
        else:
            row[column] = 0  # Handle missing values gracefully
    
    return data
data = max_absolute_scaling(data, 'Salary')

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

# 14. APRIORI STARTS HERE
import csv
from collections import defaultdict
from itertools import combinations

import pandas as pd

def read_transactions_from_csv(filename):
    df = pd.read_csv(filename, header=None, dtype=str)
    transactions = [set(df.iloc[i].dropna().str.strip()) for i in range(len(df))]
    print(transactions)
    return transactions

def has_infrequent_subset(candidate, frequent_itemsets):
    for subset in combinations(candidate, len(candidate) - 1):
        if frozenset(subset) not in frequent_itemsets:
            return False
    return True

def get_frequent_itemsets(transactions, min_support):
    item_counts = defaultdict(int)
    for transaction in transactions:
        for item in transaction:
            item_counts[frozenset([item])] += 1

    min_support_count = len(transactions) * min_support
    frequent_itemsets = {}

    for itemset, count in item_counts.items():
        if count >= min_support_count:
            frequent_itemsets[itemset] = count  # Ensure it's an integer value

    k = 2  # Ensure k is an integer
    while True:
        candidates = set()

        frequent_items = list(frequent_itemsets.keys())
        print(frequent_items)

        for i in range(len(frequent_items)):
            for j in range(i + 1, len(frequent_items)):
                items1 = set(frequent_items[i])
                items2 = set(frequent_items[j])
                union = items1.union(items2)
                if len(union) == k and has_infrequent_subset(union, frequent_itemsets):
                    candidates.add(frozenset(union))

        if not candidates:
            break

        candidate_counts = defaultdict(int)
        for transaction in transactions:
            for candidate in candidates:
                if candidate.issubset(transaction):
                    candidate_counts[candidate] += 1

        new_frequent = {}
        for itemset, count in candidate_counts.items():
            if count >= min_support_count:
                new_frequent[itemset] = count  # Ensure it's an integer value

        if not new_frequent:
            break

        frequent_itemsets.update(new_frequent)  # Ensure dictionary update is correct
        k += 1  # Ensure k is always an integer

    return frequent_itemsets


def generate_association_rules(frequent_itemsets, transactions, min_confidence):
    rules = []
    total_transactions = len(transactions)

    for itemset, support in frequent_itemsets.items():
        if len(itemset) < 2:
            continue

        for i in range(1, len(itemset)):
            for antecedent in combinations(itemset, i):
                antecedent = frozenset(antecedent)
                consequent = frozenset(itemset - antecedent)
                antecedent_support = sum(1 for t in transactions if antecedent.issubset(t))
                confidence = support / antecedent_support

                if confidence >= min_confidence:
                    support_percentage = support / total_transactions
                    rules.append((set(antecedent), set(consequent), confidence, support_percentage))

    rules.sort(key=sort_by_confidence, reverse=True)
    return rules

def sort_by_confidence(rule):
    return rule[2]

def print_rules(rules):
    """Print association rules in a readable format."""
    for antecedent, consequent, confidence, support in rules:
        print(f"{antecedent} => {consequent}")
        print(f"Confidence: {confidence:.2%}")
        print(f"Support: {support:.2%}")
        print("-" * 50)


filename = "data.csv"
min_support = 0.2  # 20% minimum support
min_confidence = 0.5  # 50% minimum confidence

transactions = read_transactions_from_csv(filename)
frequent_itemsets = get_frequent_itemsets(transactions, min_support)
print("Frequent Itemsets:")
for itemset, support in frequent_itemsets.items():
    print(f"{itemset} (Support: {support})")

print("\nAssociation Rules:")
rules = generate_association_rules(frequent_itemsets, transactions, min_confidence)
print_rules(rules)

#FP TREE
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules

def read_transactions_from_csv(filename):
    df = pd.read_csv(filename, header=None, dtype=str).fillna("")
    transactions = df.apply(lambda row: [item.strip() for item in row if item.strip()], axis=1).tolist()
    return transactions

def convert_to_dataframe(transactions):
    """Convert transactions into a one-hot encoded Pandas DataFrame."""
    unique_items = sorted(set(item for transaction in transactions for item in transaction))
    df = pd.DataFrame([{item: (item in transaction) for item in unique_items} for transaction in transactions])
    return df

filename = "data.csv"
min_support = 0.2  # Minimum support threshold (20%)
min_confidence = 0.5  # Minimum confidence threshold (50%)

# Read and process transactions
transactions = read_transactions_from_csv(filename)
df = convert_to_dataframe(transactions)

# Apply FP-Growth algorithm
frequent_itemsets = fpgrowth(df, min_support=min_support, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

# Print results
print("\nFrequent Itemsets:")
print(frequent_itemsets)

print("\nAssociation Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence']])
