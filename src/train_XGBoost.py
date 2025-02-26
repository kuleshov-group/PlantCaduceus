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
    parser.add_argument("-train", type=str, help="The directory of training data")
    parser.add_argument("-valid", type=str, help="The directory of validation data")
    parser.add_argument("-test", type=str, help="The directory of test data")
    parser.add_argument("-model", type=str, help="The directory of pre-trained model")
    parser.add_argument("-output", type=str, help="The directory of output")
    parser.add_argument("-device", type=str, default="cuda:0", help="The device to run the model")
    parser.add_argument("-batchSize", type=int, default=128, help="The batch size for the model")
    parser.add_argument("-tokenIdx", type=int, default=255, help="The index of the nucleotide")
    parser.add_argument("-test_only", action='store_true', help="Flag to perform only testing")
    parser.add_argument("-save_memory", action='store_true', help="Flag to save memory, it only works for testing")
    parser.add_argument("-chunk_size", type=int, default=100000, help="The chunk size for testing, it only works for testing when save_memory is set")
    parser.add_argument("-seed", type=int, default=42, help="The random seed to train XGBoost model")
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
    
    # Determine the appropriate dtype based on the GPU capabilities
    def get_optimal_dtype():
        if not torch.cuda.is_available():
            logging.info("Using float32 as no GPU is available.")
            return torch.float32  

        device_index = torch.cuda.current_device()
        capability = torch.cuda.get_device_capability(device_index)

        if capability[0] >= 8:  # sm_80 or higher
            logging.info("Using bfloat16 as the GPU supports sm_80 or higher.")
            return torch.bfloat16
        elif capability[0] >= 6:  # sm_60 or higher
            logging.info("Using float16 as the GPU supports sm_60 or higher.")
            return torch.float16
        else:
            logging.info("Using float32 as the GPU does not support float16 or bfloat16.")
            return torch.float32

    optimal_dtype = get_optimal_dtype()

    # Load the model with the selected dtype
    try:
        model = AutoModelForMaskedLM.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=optimal_dtype)
    except Exception as e:
        logging.error(f"Failed to load model with {optimal_dtype}, falling back to float32. Error: {e}")
        model = AutoModelForMaskedLM.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=torch.float32)

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
            token_embedding = outputs.hidden_states[-1][:, tokenIdx, :].to(torch.float32).cpu().numpy()
            embeddings.append(token_embedding)
    embeddings = np.concatenate(embeddings, axis=0)
    # average forward and reverse embeddings
    hidden_size = embeddings.shape[-1] // 2
    forward = embeddings[..., 0:hidden_size]
    reverse = embeddings[..., hidden_size:]
    reverse = reverse[..., ::-1]
    averaged_embeddings = (forward + reverse) / 2
    return averaged_embeddings

def train_xgboost_model(train_embeddings, train_labels, valid_embeddings, valid_labels, random_state=42):
    logging.info("Training XGBoost model")
    model = xgb.XGBClassifier(n_estimators=1000, max_depth=6, learning_rate=0.1, random_state=random_state, n_jobs=-1)
    model.fit(train_embeddings, train_labels, eval_set=[(valid_embeddings, valid_labels)])
    return model

def infer_xgboost_model(model, embeddings):
    logging.info("Inferencing XGBoost model")
    return model.predict_proba(embeddings)[:, 1]

def evaluate_model(predictions, labels):
    logging.info("Evaluating model")
    fpr, tpr, _ = metrics.roc_curve(labels, predictions)
    precision, recall, _ = metrics.precision_recall_curve(labels, predictions)
    roc_auc = metrics.auc(fpr, tpr)
    prauc = metrics.average_precision_score(labels, predictions)
    return fpr, tpr, precision, recall, roc_auc, prauc

def plot_metrics(fpr, tpr, precision, recall, roc_auc, prauc, output_dir, prefix='valid', random_state=42):
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
    plt.savefig(os.path.join(output_dir, f'seed_{random_state}_{prefix}_metrics.png'))
    # save metrics to file
    with open(os.path.join(output_dir, f'seed_{random_state}_{prefix}_metrics.txt'), 'w') as f:
        f.write(f"ROC AUC: {roc_auc:.2f}\n")
        f.write(f"PRAUC: {prauc:.2f}\n")


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
            xgb_model = xgb.XGBClassifier()
            xgb_model.load_model(os.path.join(args.output, f'seed_{args.seed}_XGBoost.json'))
            prefix = os.path.basename(args.test).split('.')[0]

            if args.save_memory: # split the test data into smaller chunks
                logging.info("Saving memory by splitting the test data into smaller chunks with size {}".format(args.chunk_size))
                predictions = []
                for i in range(0, len(test_sequences), args.chunk_size):
                    if os.path.exists(os.path.join(args.output, f'{prefix}_chunk_{i}_embeddings.npz')):
                        logging.info(f"Found pre-computed embeddings, loading from file {os.path.join(args.output, f'{prefix}_chunk_{i}_embeddings.npz')}")
                        embeddings = np.load(os.path.join(args.output, f'{prefix}_chunk_{i}_embeddings.npz'))
                        test_embeddings = embeddings['test']
                    else:
                        test_loader = create_dataloader(test_sequences[i:i+args.chunk_size], tokenizer, args.batchSize)
                        test_embeddings = extract_embeddings(model, test_loader, args.device, args.tokenIdx)
                        np.savez_compressed(os.path.join(args.output, f'{prefix}_chunk_{i}_embeddings.npz'), test=test_embeddings)
                    pred = infer_xgboost_model(xgb_model, test_embeddings)
                    pred = pred[:, np.newaxis]
                    predictions.extend(pred)
                predictions = np.concatenate(predictions, axis=0)
            else:
                if os.path.exists(os.path.join(args.output, f'{prefix}_embeddings.npz')):
                    logging.info(f"Found pre-computed embeddings, loading from file {os.path.join(args.output, f'{prefix}_embeddings.npz')}")
                    embeddings = np.load(os.path.join(args.output, f'{prefix}_embeddings.npz'))
                    test_embeddings = embeddings['test']
                else:
                    test_loader = create_dataloader(test_sequences, tokenizer, args.batchSize)
                    test_embeddings = extract_embeddings(model, test_loader, args.device, args.tokenIdx)
                    np.savez_compressed(os.path.join(args.output, prefix + '_embeddings.npz'), test=test_embeddings)
                predictions = infer_xgboost_model(xgb_model, test_embeddings)
                    
            np.savez_compressed(os.path.join(args.output, f'seed_{args.seed}_{prefix}_predictions.npz'), predictions=predictions)
            fpr, tpr, precision, recall, roc_auc, prauc = evaluate_model(predictions, test_labels)
            plot_metrics(fpr, tpr, precision, recall, roc_auc, prauc, args.output, prefix=prefix, random_state=args.seed)
        else:
            logging.error("Please provide the test data")
    else:
        train_sequences, train_labels = load_data(args.train)
        valid_sequences, valid_labels = load_data(args.valid)

        train_loader = create_dataloader(train_sequences, tokenizer, args.batchSize)
        valid_loader = create_dataloader(valid_sequences, tokenizer, args.batchSize)
        if os.path.exists(os.path.join(args.output, 'train_valid_embeddings.npz')):
            logging.info(f"Found pre-computed embeddings, loading from file {os.path.join(args.output, 'train_valid_embeddings.npz')}")
            embeddings = np.load(os.path.join(args.output, 'train_valid_embeddings.npz'))
            train_embeddings = embeddings['train']
            valid_embeddings = embeddings['valid']
        else:
            train_embeddings = extract_embeddings(model, train_loader, args.device, args.tokenIdx)
            valid_embeddings = extract_embeddings(model, valid_loader, args.device, args.tokenIdx)
            np.savez_compressed(os.path.join(args.output, 'train_valid_embeddings.npz'), train=train_embeddings, valid=valid_embeddings)

        if os.path.exists(os.path.join(args.output, f'seed_{args.seed}_XGBoost.json')):
            logging.info(f"Found pre-trained XGBoost model, loading from file {os.path.join(args.output, f'seed_{args.seed}_XGBoost.json')}")
            xgb_model = xgb.XGBClassifier()
            xgb_model.load_model(os.path.join(args.output, f'seed_{args.seed}_XGBoost.json'))
        else:
            xgb_model = train_xgboost_model(train_embeddings, train_labels, valid_embeddings, valid_labels, random_state=args.seed)
            xgb_model.save_model(os.path.join(args.output, f'seed_{args.seed}_XGBoost.json'))
            valid_predictions = infer_xgboost_model(xgb_model, valid_embeddings)
            np.savez_compressed(os.path.join(args.output, f'seed_{args.seed}_valid_predictions.npz'), predictions=valid_predictions)
            fpr, tpr, precision, recall, roc_auc, prauc = evaluate_model(valid_predictions, valid_labels)
            prefix = os.path.basename(args.valid).split('.')[0]
            plot_metrics(fpr, tpr, precision, recall, roc_auc, prauc, args.output, prefix=prefix, random_state=args.seed)

        if args.test:
            test_sequences, test_labels = load_data(args.test)
            xgb_model = xgb.XGBClassifier()
            xgb_model.load_model(os.path.join(args.output, f'seed_{args.seed}_XGBoost.json'))
            prefix = os.path.basename(args.test).split('.')[0]

            if args.save_memory: # split the test data into smaller chunks
                logging.info("Saving memory by splitting the test data into smaller chunks with size {}".format(args.chunk_size))
                predictions = []
                for i in range(0, len(test_sequences), args.chunk_size):
                    if os.path.exists(os.path.join(args.output, f'{prefix}_chunk_{i}_embeddings.npz')):
                        logging.info(f"Found pre-computed embeddings, loading from file {os.path.join(args.output, f'{prefix}_chunk_{i}_embeddings.npz')}")
                        embeddings = np.load(os.path.join(args.output, f'{prefix}_chunk_{i}_embeddings.npz'))
                        test_embeddings = embeddings['test']
                    else:
                        test_loader = create_dataloader(test_sequences[i:i+args.chunk_size], tokenizer, args.batchSize)
                        test_embeddings = extract_embeddings(model, test_loader, args.device, args.tokenIdx)
                        np.savez_compressed(os.path.join(args.output, f'{prefix}_chunk_{i}_embeddings.npz'), test=test_embeddings)
                    pred = infer_xgboost_model(xgb_model, test_embeddings)
                    pred = pred[:, np.newaxis]
                    predictions.extend(pred)
                predictions = np.concatenate(predictions, axis=0)
            else:
                if os.path.exists(os.path.join(args.output, f'{prefix}_embeddings.npz')):
                    logging.info(f"Found pre-computed embeddings, loading from file {os.path.join(args.output, f'{prefix}_embeddings.npz')}")
                    embeddings = np.load(os.path.join(args.output, f'{prefix}_embeddings.npz'))
                    test_embeddings = embeddings['test']
                else:
                    test_loader = create_dataloader(test_sequences, tokenizer, args.batchSize)
                    test_embeddings = extract_embeddings(model, test_loader, args.device, args.tokenIdx)
                    np.savez_compressed(os.path.join(args.output, prefix + '_embeddings.npz'), test=test_embeddings)
                predictions = infer_xgboost_model(xgb_model, test_embeddings)
                    
            np.savez_compressed(os.path.join(args.output, f'seed_{args.seed}_{prefix}_predictions.npz'), predictions=predictions)
            fpr, tpr, precision, recall, roc_auc, prauc = evaluate_model(predictions, test_labels)
            prefix = os.path.basename(args.test).split('.')[0]
            plot_metrics(fpr, tpr, precision, recall, roc_auc, prauc, args.output, prefix=prefix, random_state=args.seed)

if __name__ == "__main__":
    main()
