import argparse, os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer
from tqdm import tqdm
import xgboost as xgb
from sklearn import metrics
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-train", type=str, help="The directory of input fasta")
    parser.add_argument("-valid", type=str, help="The directory of output")
    parser.add_argument("-test", type=str, help="The directory of test fasta")
    parser.add_argument("-model", type=str, help="The directory of pre-trained model")
    parser.add_argument("-output", type=str, help="The directory of output")
    parser.add_argument("-device", type=str, default="cuda:0", help="The device to run the model")
    parser.add_argument("-batchSize", type=int, default=128, help="The batch size for the model")
    parser.add_argument("-tokenIdx", type=int, default=255, help="The index of the nucleotide")
    return parser.parse_args()

class SequenceDataset(Dataset):
    def __init__(self, sequences, tokenizer):
        self.sequences = sequences
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        encoding = self.tokenizer.encode_plus(
            sequence,
            return_tensors="pt",
            return_attention_mask=False,
            return_token_type_ids=False
        )
        input_ids = encoding['input_ids']
        return {
            'sequence': sequence,
            'input_ids': input_ids.squeeze()
        }

def load_data(filepath):
    data = pd.read_csv(filepath, delimiter='\t')
    return data['sequences'].tolist(), data['label'].tolist()

def create_dataloader(sequences, tokenizer, batch_size):
    dataset = SequenceDataset(sequences, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def load_model_and_tokenizer(model_dir, device):
    model = AutoModelForMaskedLM.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model.to(device)
    return model, tokenizer

def extract_embeddings(model, dataloader, device, tokenIdx):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting Embeddings"):
            input_ids = batch['input_ids'].to(device)
            with torch.inference_mode():
                outputs = model(input_ids=input_ids, output_hidden_states=True)
            token_embedding = outputs.hidden_states[-1][:, tokenIdx, :].cpu().numpy()
            embeddings.append(token_embedding)
    embeddings = np.concatenate(embeddings, axis=0)
    # average forward and reverse embeddings
    hidden_size = embeddings.shape[-1] // 2
    forward = embeddings[..., 0:hidden_size]
    reverse = embeddings[..., hidden_size:]
    reverse = reverse[..., ::-1]
    averaged_embeddings = (forward + reverse) / 2
    return averaged_embeddings

def train_xgboost_model(train_embeddings, train_labels, valid_embeddings, valid_labels):
    model = xgb.XGBClassifier(n_estimators=1000, max_depth=6, learning_rate=0.1, n_jobs=64)
    model.fit(train_embeddings, train_labels, eval_set=[(valid_embeddings, valid_labels)])
    return model

def evaluate_model(model, embeddings, labels):
    predictions = model.predict_proba(embeddings)[:, 1]
    fpr, tpr, _ = metrics.roc_curve(labels, predictions)
    precision, recall, _ = metrics.precision_recall_curve(labels, predictions)
    roc_auc = metrics.auc(fpr, tpr)
    prauc = metrics.average_precision_score(labels, predictions)
    return fpr, tpr, precision, recall, roc_auc, prauc

def plot_metrics(fpr, tpr, precision, recall, roc_auc, prauc, output_dir, prefix='valid'):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}', linewidth=2)
    axs[0].set_title('ROC Curve')
    axs[0].set_xlabel('False Positive Rate')
    axs[0].set_ylabel('True Positive Rate')
    axs[0].legend(loc='lower right')

    axs[1].plot(recall, precision, label=f'PRAUC = {prauc:.2f}', linewidth=2)
    axs[1].set_title('Precision-Recall Curve')
    axs[1].set_xlabel('Recall')
    axs[1].set_ylabel('Precision')
    axs[1].legend(loc='lower left')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, prefix + '_metrics.png'))

def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    train_sequences, train_labels = load_data(args.train)
    valid_sequences, valid_labels = load_data(args.valid)

    model, tokenizer = load_model_and_tokenizer(args.model, args.device)

    train_loader = create_dataloader(train_sequences, tokenizer, args.batchSize)
    valid_loader = create_dataloader(valid_sequences, tokenizer, args.batchSize)

    train_embeddings = extract_embeddings(model, train_loader, args.device, args.tokenIdx)
    valid_embeddings = extract_embeddings(model, valid_loader, args.device, args.tokenIdx)

    xgb_model = train_xgboost_model(train_embeddings, train_labels, valid_embeddings, valid_labels)
    np.save(os.path.join(args.output, 'model.json'), xgb_model)
    fpr, tpr, precision, recall, roc_auc, prauc = evaluate_model(xgb_model, valid_embeddings, valid_labels)
    plot_metrics(fpr, tpr, precision, recall, roc_auc, prauc, args.output, prefix='valid')

    if args.test:
        test_sequences, test_labels = load_data(args.test)
        test_loader = create_dataloader(test_sequences, tokenizer, args.batchSize)
        test_embeddings = extract_embeddings(model, test_loader, args.device, args.tokenIdx)
        fpr, tpr, precision, recall, roc_auc, prauc = evaluate_model(xgb_model, test_embeddings, test_labels)
        plot_metrics(fpr, tpr, precision, recall, roc_auc, prauc, args.output, prefix='test')

if __name__ == "__main__":
    main()