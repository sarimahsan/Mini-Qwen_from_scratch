import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

class HuggingFaceDataset(Dataset):
    """
    A PyTorch Dataset that loads a text dataset from Hugging Face,
    tokenizes it using a pretrained tokenizer, and chunks it into
    sequences of seq_len for causal language modeling.
    """
    def __init__(self, dataset_name="wikitext", dataset_config="wikitext-2-raw-v1", split="train", tokenizer_name="gpt2", seq_len=128):
        print(f"Loading dataset '{dataset_name}' ({dataset_config}) from Hugging Face...")
        # Load the dataset
        raw_dataset = load_dataset(dataset_name, dataset_config, split=split)
        
        print(f"Loading tokenizer '{tokenizer_name}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print("Tokenizing dataset...")
        # Filter out empty lines to save memory and speed up
        texts = [text for text in raw_dataset["text"] if text.strip()]
        
        # Concatenate text with end-of-text tokens
        full_text = self.tokenizer.eos_token.join(texts)
        
        # Tokenize the entire corpus
        self.tokens = self.tokenizer.encode(full_text)
        self.seq_len = seq_len
        
        print(f"Tokenization complete. Total tokens: {len(self.tokens)}")

    def __len__(self):
        # We need seq_len tokens for input and 1 shifted token for label
        return (len(self.tokens) - 1) // self.seq_len

    def __getitem__(self, idx):
        start_idx = idx * self.seq_len
        end_idx = start_idx + self.seq_len
        
        # Extract chunk of size seq_len + 1
        chunk = self.tokens[start_idx:end_idx + 1]
        
        # x is the sequence, y is the sequence shifted by 1
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

def get_dataloader(dataset_name="wikitext", dataset_config="wikitext-2-raw-v1", split="train", tokenizer_name="gpt2", seq_len=128, batch_size=8, shuffle=True):
    dataset = HuggingFaceDataset(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split=split,
        tokenizer_name=tokenizer_name,
        seq_len=seq_len
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True
    )
    return loader, dataset.tokenizer
