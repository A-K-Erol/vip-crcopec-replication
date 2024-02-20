import pandas as pd

# sampling algorithm
full_train_df = pd.read_csv("data/cr-copec_total_train.csv")
train_yes = full_train_df[full_train_df['true_label'] == 'yes']
train_no = full_train_df[full_train_df['true_label'] == 'no']

full_val_df = pd.read_csv("data/cr-copec_total_val.csv")
val_yes = full_val_df[full_val_df['true_label'] == 'yes']
val_no = full_val_df[full_val_df['true_label'] == 'no']

test_df = pd.read_csv("data/cr-copec_total_test.csv")

# sample_train = pd.concat([train_yes.sample(2000), train_no.sample(3000)])
sample_train = pd.concat([full_train_df.sample(6000)])
sample_val = pd.concat([val_yes.sample(800), val_no.sample(800)])
test_df = test_df.sample(800)

train_texts, train_labels = sample_train.sentence.values.tolist(), [1 if i=='yes' else 0 for i in sample_train.true_label.values]
val_texts, val_labels = sample_val.sentence.values.tolist(), [1 if i=='yes' else 0 for i in sample_val.true_label.values]
test_texts, test_labels = test_df.sentence.values.tolist(), [1 if i=='yes' else 0 for i in test_df.true_label.values]
print(len(train_texts), len(val_texts), len(test_texts))

# Load the datasets into a pandas dataframe.

# Custom Data for Training and Validation
train_df = pd.read_csv("data/cr-copec_total_train.csv", nrows=2000)
val_df = pd.read_csv("data/cr-copec_total_val.csv", nrows=500)
test_df = pd.read_csv("data/cr-copec_total_test.csv", nrows=500)

train_texts, train_labels = train_df.sentence.values.tolist(), [1 if i=='yes' else 0 for i in train_df.true_label.values]
val_texts, val_labels = val_df.sentence.values.tolist(), [1 if i=='yes' else 0 for i in val_df.true_label.values]
test_texts, test_labels = test_df.sentence.values.tolist(), [1 if i=='yes' else 0 for i in test_df.true_label.values]

# Full Data Always Available for Testing
full_test_df = pd.read_csv("data/cr-copec_total_test.csv")
full_test_texts, full_test_labels = full_test_df.sentence.values.tolist(), [1 if i=='yes' else 0 for i in full_test_df.true_label.values]


# Sampled Partial Dataset
sampled_train_df = pd.concat([chunk.iloc[0] for chunk in pd.read_csv("data/cr-copec_total_train.csv", skiprows=range(1, 300), chunksize=301)])
sampled_val_df = pd.concat([chunk.iloc[0] for chunk in pd.read_csv("data/cr-copec_total_val.csv", skiprows=range(1, 200), chunksize=201)])
sampled_train_texts, sampled_train_labels = sampled_train_df.sentence.values.tolist(), [1 if i=='yes' else 0 for i in sampled_train_df.true_label.values]
sampled_val_texts, sampled_val_labels = sampled_val_df.sentence.values.tolist(), [1 if i=='yes' else 0 for i in sampled_val_df.true_label.values]

sampled_test_df = pd.concat([chunk.iloc[0] for chunk in pd.read_csv("data/cr-copec_total_test.csv", skiprows=range(1, 10), chunksize=11)])
sampled_test_texts, sampled_test_labels = sampled_test_df.sentence.values.tolist(), [1 if i=='yes' else 0 for i in sampled_test_df.true_label.values]

print(len(sampled_test_texts))
print(len(sampled_train_texts))
print(len(sampled_train_labels))
print(len(sampled_test_labels))
# Report the number of sentences.
print('Number of custom training sentences: {:,}\n'.format(train_df.shape[0]))
print('Number of custom validation sentences: {:,}\n'.format(val_df.shape[0]))
print('Number of custom test sentences: {:,}\n'.format(test_df.shape[0]))
print('Number of full test sentences: {:,}\n'.format(full_test_df.shape[0]))
print('Number of sampled train sentences: {:,}\n'.format(sampled_train_df.shape[0]))
print('Number of sampled validation sentences: {:,}\n'.format(sampled_val_df.shape[0]))
print('Number of sampled test sentences: {:,}\n'.format(sampled_train_df.shape[0]))