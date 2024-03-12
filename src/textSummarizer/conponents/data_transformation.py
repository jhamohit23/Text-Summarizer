import os
from textSummarizer.logging import logger
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk

import os
from transformers import AutoTokenizer
from datasets import load_from_disk

class DataTransformation:
    def __init__(self, config):
        self.config = config
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        except Exception as e:
            print("Error initializing tokenizer:", e)
            # Handle the error as per your application's requirements

    
    def convert_examples_to_features(self, example_batch):
        try:
            input_encodings = self.tokenizer(example_batch['dialogue'], max_length=1024, truncation=True)
            with self.tokenizer.as_target_tokenizer():
                target_encodings = self.tokenizer(example_batch['summary'], max_length=128, truncation=True)
        except Exception as e:
            print("Error converting examples to features:", e)
            return None
        
        return {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': target_encodings['input_ids']
        }
    

    def convert(self):
        try:
            dataset_samsum = load_from_disk(self.config.data_path)
            dataset_samsum_pt = dataset_samsum.map(self.convert_examples_to_features, batched=True)
            dataset_samsum_pt.save_to_disk(os.path.join(self.config.root_dir, "samsum_dataset"))
        except Exception as e:
            print("Error converting dataset:", e)
            # Handle the error as per your application's requirements

# Usage
class DataTransformationConfig:
    def __init__(self, tokenizer_name, data_path, root_dir):
        self.tokenizer_name = tokenizer_name
        self.data_path = data_path
        self.root_dir = root_dir

config = DataTransformationConfig(
    tokenizer_name="google/pegasus-cnn_dailymail",
    data_path="path/to/your/data",
    root_dir="path/to/save/processed/data"
)

data_transformer = DataTransformation(config)
data_transformer.convert()
