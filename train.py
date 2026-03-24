from email.parser import BytesParser
from email import policy
import os
from tqdm import tqdm
import pandas as pd
from torch.utils.data import DataLoader
from bert.bert_classfifer import BertClassifier, TextDataset
from load_emails import load_emails

email_path = "data/TRAINING/"

rows = load_emails(email_path, "data/SPAMTrain.label")

df = pd.DataFrame(rows, columns=['text', 'labels'])


bc = BertClassifier()

print('creating dataset and loader....')
data = TextDataset(list(df['text']), df['labels'].astype(int), bc.tokenizer)
loader = DataLoader(data, 32, True)
print("Done!")

bc.fit(
   loader=loader
)
