import urllib.request
import zipfile
import os
from pathlib import Path
import torch
import pandas as pd
from torch.utils.data import Dataset

device = "cuda" if torch.cuda.is_available() else "cpu"

url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_path = "sms_spam_collection.zip"
extracted_path = "sms_spam_collection"
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download and extraction.")
        return
    
    with urllib.request.urlopen(url) as response: # Open the URL and read the response
        with open(zip_path, 'wb') as out_file:
            out_file.write(response.read())
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref: # Open the downloaded zip file and extract its contents
        zip_ref.extractall(extracted_path)
        
    original_file_path = Path(extracted_path) / "SMSSpamCollection" # The original file name inside the zip
    os.rename(original_file_path, data_file_path)
    print(f"File downloaded and saved as {data_file_path}")

def create_balanced_dataset(df):
    num_spam = df[df["Label"] == "spam"].shape[0]
    ham_sampled = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
    balanced_df = pd.concat([ham_sampled, df[df["Label"] == "spam"]])
    return balanced_df

def random_split(df, train_frac, validation_frac):
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)  # Shuffle the DataFrame
    
    train_end = int(train_frac * len(df)) # Calculate the index for the end of the training set
    val_end = train_end + int(validation_frac * len(df)) # Calculate the index for the end of the validation set
    
    train_df = df[:train_end] # Get the training set
    val_df = df[train_end:val_end] # Get the validation set
    test_df = df[val_end:] # Get the test set
    
    return train_df, val_df, test_df


class SpamDataset(Dataset):
    
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):

        self.data = pd.read_csv(csv_file)
        
        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]]
        
        if max_length is None:
            self.max_length = self._longest_encoded_length()  # Determine the maximum length based on the longest encoded text
        
        else: 
            self.max_length = max_length  # Use the provided max_length
            
            self.encoded_text = [  
                encoded_text[:self.max_length]  # Truncate the encoded text to the specified max_length
                for encoded_text in self.encoded_texts
            ]
        
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text)) 
            for encoded_text in self.encoded_texts 
        ]
            # Pad the encoded texts to the max_length using the pad_token_id
        
    def __getitem__(self, index): 
        encoded = self.encoded_texts[index] # Get the encoded text for the given index
        label = self.data.iloc[index]["Label"] # Get the label for the given index
        return (
            torch.tensor(encoded, dtype=torch.long), 
            torch.tensor(label, dtype=torch.long)
        )
        
    def __len__(self):
        return len(self.data)
    
    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length: # Update max_length if the current encoded text is longer
                max_length = encoded_length
        return max_length