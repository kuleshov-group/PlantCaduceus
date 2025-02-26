import argparse, os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from train_XGBoost import *
import logging

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-test", type=str, help="The directory of test data")
    parser.add_argument("-model", type=str, help="The directory of pre-trained model")
    parser.add_argument("-classifier", type=str, help="The directory of trained XGBoost models")
    parser.add_argument("-output", type=str, help="The directory of output")
    parser.add_argument("-device", type=str, default="cuda:0", help="The device to run the model")
    parser.add_argument("-batchSize", type=int, default=128, help="The batch size for the model")
    parser.add_argument("-tokenIdx", type=int, default=255, help="The index of the nucleotide")
    parser.add_argument("-save_memory", action='store_true', help="Flag to save memory, it only works for testing")
    parser.add_argument("-chunk_size", type=int, default=100000, help="The chunk size for testing, it only works for testing when save_memory is set")
    return parser.parse_args()

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(args.model, args.device)

    test_sequences, test_labels = load_data(args.test)
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(args.classifier)
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

    output_df = pd.DataFrame({
        'label': test_labels,
        'prediction': predictions
    })
    output_df.to_csv(os.path.join(args.output, f'{prefix}_predictions.tsv'), sep='\t', index=False)
    logging.info(f"Saved predictions to {os.path.join(args.output, f'{prefix}_predictions.tsv')}")

if __name__ == "__main__":
    main()
