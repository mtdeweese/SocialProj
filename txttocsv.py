import pandas as pd


df = pd.read_csv('data/train.txt', sep='\t') 


df['content'] = df['title'].fillna('') + ' ' + df['text'].fillna('')

# Select only the desired columns
df_new = df[['company', 'content', 'label']]

# Save as CSV
df_new.to_csv('train.csv', index=False)


df = pd.read_csv('data/test.txt', sep='\t') 


df['content'] = df['title'].fillna('') + ' ' + df['text'].fillna('')

# Select only the desired columns
df_new = df[['company', 'content', 'label']]

# Save as CSV
df_new.to_csv('test.csv', index=False)