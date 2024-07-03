from transformers import AutoTokenizer, AutoConfig, AutoModelForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, EarlyStoppingCallback, AutoConfig
from torch.utils.data import Dataset
import torch
import pandas as pd
import matplotlib.pyplot as plt
from transformers.models.bert.modeling_bert import BertEmbeddings
import torch.nn as nn

#TODO refactor once other PRs are merged in
#TODO early version

#TODO move to data folder
class DNADataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings["input_ids"])

# TODO generalize
class CustomBertEmbeddings(BertEmbeddings):
    def __init__(self, config):
        super().__init__(config)
        self.word_embeddings = nn.Embedding(7, config.hidden_size, padding_idx=config.pad_token_id)
tokenizer = AutoTokenizer.from_pretrained("gonzalobenegas/tokenizer-dna-mlm")

config = AutoConfig.from_pretrained('mosaicml/mosaic-bert-base')
config.vocab_size = len(tokenizer.get_vocab())
# TODO swap out model architecture for MosaicBERT
# TODO implement unpadding
# Improve training strategy /for mosaic BERT
#model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased', cols nfig = config, ignore_mismatched_sizes=True)
# Unpadding is not needed... All our sequences are soooo long.
model = AutoModelForMaskedLM.from_pretrained("mosaicml/mosaic-bert-base", 
                                             config=config, 
                                             ignore_mismatched_sizes=True, 
                                             trust_remote_code=True,
                                             revision="b5cb8d59")
model.bert.embeddings = CustomBertEmbeddings(config)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model.to(device)

texts = pd.read_csv(filepath_or_buffer='../BERT/00_train.tsv', delimiter='\t')
texts.head()

texts = texts['seq'].tolist()

encodings = tokenizer(texts[0:1000], truncation=True, padding=True)

dataset = DNADataset(encodings)

training_args = TrainingArguments(
    output_dir="./DNABert",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    save_steps=500,
    save_total_limit=20,
    gradient_accumulation_steps=25,
    adam_epsilon=1e-6,
    weight_decay=0.01,
    adam_beta1=0.9,
    adam_beta2=0.98,
    warmup_steps = 2000,
    max_steps = 8000,
    learning_rate = 4e-4,
    logging_steps=10,
    load_best_model_at_end=True,
    save_strategy="steps",
    evaluation_strategy="steps"
)

# Replace with MosaicBERT version
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
    eval_dataset=dataset
)

trainer.train()
