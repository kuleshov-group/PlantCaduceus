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
import logging

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
    parser.add_argument("-test_only", action='store_true', help="Flag to perform only testing")
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
    logging.info(f"Loading data from {filepath}")
    data = pd.read_csv(filepath, delimiter='\t')
    return data['sequences'].tolist(), data['label'].tolist()

def create_dataloader(sequences, tokenizer, batch_size):
    logging.info(f"Creating DataLoader with batch size {batch_size}")
    dataset = SequenceDataset(sequences, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

def load_model_and_tokenizer(model_dir, device):
    logging.info(f"Loading model and tokenizer from {model_dir}")
    try:
        model = AutoModelForMaskedLM.from_pretrained(model_dir, trust_remote_code=True, dtype=torch.bfloat16)
    except:
        model = AutoModelForMaskedLM.from_pretrained(model_dir, trust_remote_code=True, dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model.to(device)
    return model, tokenizer

def extract_embeddings(model, dataloader, device, tokenIdx):
    logging.info("Extracting embeddings")
    model.eval()
    embeddings = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
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
    logging.info("Training XGBoost model")
    model = xgb.XGBClassifier(n_estimators=1000, max_depth=6, learning_rate=0.1, n_jobs=64)
    model.fit(train_embeddings, train_labels, eval_set=[(valid_embeddings, valid_labels)])
    return model

def evaluate_model(model, embeddings, labels):
    logging.info("Evaluating model")
    predictions = model.predict_proba(embeddings)[:, 1]
    fpr, tpr, _ = metrics.roc_curve(labels, predictions)
    precision, recall, _ = metrics.precision_recall_curve(labels, predictions)
    roc_auc = metrics.auc(fpr, tpr)
    prauc = metrics.average_precision_score(labels, predictions)
    return fpr, tpr, precision, recall, roc_auc, prauc

def plot_metrics(fpr, tpr, precision, recall, roc_auc, prauc, output_dir, prefix='valid'):
    logging.info(f"Plotting metrics and saving to {output_dir}")
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
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(args.model, args.device)

    if args.test_only:
        if args.test:
            test_sequences, test_labels = load_data(args.test)
            test_loader = create_dataloader(test_sequences, tokenizer, args.batchSize)
            test_embeddings = extract_embeddings(model, test_loader, args.device, args.tokenIdx)
            prefix = os.path.basename(args.test).split('.')[0]
            np.savez_compressed(os.path.join(args.output, prefix + '_embeddings.npz'), test=test_embeddings)
            
            xgb_model = xgb.XGBClassifier()
            xgb_model.load_model(os.path.join(args.output, 'model.json'))
            
            fpr, tpr, precision, recall, roc_auc, prauc = evaluate_model(xgb_model, test_embeddings, test_labels)
            prefix = os.path.basename(args.test).split('.')[0]
            plot_metrics(fpr, tpr, precision, recall, roc_auc, prauc, args.output, prefix=prefix)
    else:
        train_sequences, train_labels = load_data(args.train)
        valid_sequences, valid_labels = load_data(args.valid)

        train_loader = create_dataloader(train_sequences, tokenizer, args.batchSize)
        valid_loader = create_dataloader(valid_sequences, tokenizer, args.batchSize)

        train_embeddings = extract_embeddings(model, train_loader, args.device, args.tokenIdx)
        valid_embeddings = extract_embeddings(model, valid_loader, args.device, args.tokenIdx)
        np.savez_compressed(os.path.join(args.output, 'train_valid_embeddings.npz'), train=train_embeddings, valid=valid_embeddings)

        xgb_model = train_xgboost_model(train_embeddings, train_labels, valid_embeddings, valid_labels)
        xgb_model.save_model(os.path.join(args.output, 'model.json'))
        fpr, tpr, precision, recall, roc_auc, prauc = evaluate_model(xgb_model, valid_embeddings, valid_labels)
        prefix = os.path.basename(args.valid).split('.')[0]
        plot_metrics(fpr, tpr, precision, recall, roc_auc, prauc, args.output, prefix=prefix)

        if args.test:
            test_sequences, test_labels = load_data(args.test)
            test_loader = create_dataloader(test_sequences, tokenizer, args.batchSize)
            test_embeddings = extract_embeddings(model, test_loader, args.device, args.tokenIdx)
            prefix = os.path.basename(args.test).split('.')[0]
            np.savez_compressed(os.path.join(args.output, prefix + '_embeddings.npz'), test=test_embeddings)
            fpr, tpr, precision, recall, roc_auc, prauc = evaluate_model(xgb_model, test_embeddings, test_labels)
            prefix = os.path.basename(args.test).split('.')[0]
            plot_metrics(fpr, tpr, precision, recall, roc_auc, prauc, args.output, prefix=prefix)

if __name__ == "__main__":
    main()
