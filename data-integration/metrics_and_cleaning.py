import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

integrated_dataset = pd.read_csv('integrated_datasets/category_annotation.csv')


# Calculation of precision, recall and f1-score metrics
original_categories = []
reconciled_categories = []

for i in range(1, 8):
    original_category = integrated_dataset[f'categoria_{i}']
    reconciled_category = integrated_dataset[f'categoria_{i}_riconciliata']
    original_categories.extend(original_category)
    reconciled_categories.extend(reconciled_category)

precision = precision_score(original_categories, reconciled_categories, average='macro', zero_division=0)
recall = recall_score(original_categories, reconciled_categories, average='macro', zero_division=0)
f1 = f1_score(original_categories, reconciled_categories, average='macro', zero_division=0)

print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-score: {f1}')


# Category columns cleaning
integrated_dataset.drop(columns=['categoria_1', 'categoria_2', 'categoria_3', 'categoria_4',
                                 'categoria_5', 'categoria_6', 'categoria_7'], inplace=True)
integrated_dataset.columns = integrated_dataset.columns.str.replace('_riconciliata', '')

categories = []

for row in range(len(integrated_dataset)):
    for i in range(1, 8):
        if integrated_dataset.at[row, f'categoria_{i}'] not in categories:
            categories.append(integrated_dataset.at[row, f'categoria_{i}'])
        else:
            integrated_dataset.at[row, f'categoria_{i}'] = ''
    categories.clear()

integrated_dataset.to_csv('integrated_datasets/category_annotation_cleaned.csv', sep=';', index=False)