#!/usr/bin/env python3
"""
Zero-shot evaluation utilities for PlantCAD2.
"""

import json
import logging
from typing import List, Optional, Sequence

import fire
import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from datasets import load_dataset
from sklearn.metrics import roc_curve, auc, average_precision_score
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer
from tqdm import tqdm


logger = logging.getLogger(__name__)

# Canonical nucleotide ordering and helpers
NUCLEOTIDES = ("A", "C", "G", "T")
NUCLEOTIDES_LOWER = tuple(n.lower() for n in NUCLEOTIDES)
NUCLEOTIDE_TO_INDEX = {b: i for i, b in enumerate(NUCLEOTIDES)}


def _require_cuda() -> None:
    """Validate that CUDA is available.

    This script requires a CUDA-capable GPU. CPU execution is not supported to avoid
    unintended slowdowns or dtype/device mismatches.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for zero-shot evaluation; CPU is not supported. "
            "Ensure a CUDA GPU is available."
        )
    n = torch.cuda.device_count()
    logger.info("Using %d GPU(s) for inference.", n)


def _optimal_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    major, _ = torch.cuda.get_device_capability(torch.cuda.current_device())
    if major >= 8:
        return torch.bfloat16
    if major >= 6:
        return torch.float16
    return torch.float32


def _load_model(model_name: str):
    dtype = _optimal_dtype()
    logger.warning(
        "Loading model with dtype %s (no auto-fallback). Incompatible weights or kernels may error.",
        dtype,
    )
    model = AutoModelForMaskedLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=dtype
    )
    # Note: We set dtype in two places because some transformers versions are inconsistent in
    # honoring torch_dtype during from_pretrained. See:
    # https://github.com/huggingface/transformers/issues/36567
    # Passing torch_dtype here and also calling model.to(dtype) below ensures correctness across
    # versions; when from_pretrained respects torch_dtype, the subsequent .to(dtype) is a no-op.
    # model.to(dtype)
    model.to(torch.float32)  # temporary due to hardware compatibility issues
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model.eval()
    return model, tok


class SingleMaskDataset(Dataset):
    def __init__(self, sequences: pd.Series, tokenizer, token_idx: int):
        self.sequences = sequences.reset_index(drop=True)
        self.tok = tokenizer
        self.idx = token_idx

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, i: int):
        seq = self.sequences.iloc[i]
        enc = self.tok(
            seq,
            return_tensors="pt",
            padding=False,
            truncation=False,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        input_ids = enc["input_ids"]
        assert input_ids.size(1) > self.idx, (
            f"token_idx {self.idx} out of range for sequence length {input_ids.size(1)}"
        )
        input_ids[0, self.idx] = self.tok.mask_token_id
        return {"masked_ids": input_ids}


class MultiMaskDataset(Dataset):
    def __init__(self, sequences: pd.Series, tokenizer, mask_idx: List[int]):
        self.sequences = sequences.reset_index(drop=True)
        self.tok = tokenizer
        self.mask_idx = mask_idx

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, i: int):
        seq = self.sequences.iloc[i]
        enc = self.tok(
            seq,
            return_tensors="pt",
            padding=False,
            truncation=False,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        input_ids = enc["input_ids"]
        assert input_ids.size(1) > max(self.mask_idx), "mask index out of range"
        input_ids[0, self.mask_idx] = self.tok.mask_token_id
        return {"masked_ids": input_ids}


class UnmaskedDataset(Dataset):
    def __init__(self, sequences: pd.Series, tokenizer):
        self.seqs = sequences.astype(str).tolist()
        self.tok = tokenizer

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i: int):
        enc = self.tok(
            self.seqs[i],
            truncation=False,
            padding=False,
            return_tensors="pt",
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return {"input_ids": enc["input_ids"].squeeze(0)}


def _masked_probs(model, tokenizer, loader, accelerator: Accelerator, desc: str = "Masked logits") -> np.ndarray:
    idxs = [tokenizer.get_vocab()[n] for n in NUCLEOTIDES_LOWER]
    all_probs = []
    for batch in tqdm(loader, desc=desc, disable=not accelerator.is_local_main_process):
        cur_ids = batch["masked_ids"].squeeze(1)
        with torch.inference_mode():
            logits = model(input_ids=cur_ids).logits
        masked_pos = (cur_ids == tokenizer.mask_token_id).unsqueeze(-1).expand(-1, -1, logits.size(-1))
        masked_logits = torch.masked_select(logits, masked_pos).view(-1, logits.size(-1))
        probs = torch.softmax(masked_logits[:, idxs].float(), dim=-1)
        all_probs.append(probs)
    all_probs = torch.cat(all_probs, dim=0)
    return accelerator.gather_for_metrics(all_probs).cpu().numpy()


def _unmasked_probs(loader, model, tokenizer, accelerator: Accelerator, desc: str = "Inference (unmasked)") -> np.ndarray:
    """Per-position probabilities over A,C,G,T for each sequence. Shape: [N, L, 4]."""
    idxs = [tokenizer.get_vocab()[n] for n in NUCLEOTIDES_LOWER]
    first_len = None
    all_probs = []
    for batch in tqdm(loader, desc=desc, disable=not accelerator.is_local_main_process):
        input_ids = batch["input_ids"]
        with torch.inference_mode():
            out = model(input_ids=input_ids)
        logits = out.logits[..., idxs]
        probs = torch.softmax(logits.float(), dim=-1)
        if first_len is None:
            first_len = probs.shape[1]
        elif probs.shape[1] != first_len:
            raise ValueError(f"All sequences must have same length; got {probs.shape[1]} vs {first_len}")
        all_probs.append(probs)
    all_probs = torch.cat(all_probs, dim=0)          # [N_local, L, 4]
    gathered = accelerator.gather_for_metrics(all_probs)  # [N_total, L, 4]
    return gathered.cpu().numpy()


def _sv_llr_boundary(
    df: pd.DataFrame,
    ref_probs: np.ndarray,
    mut_probs: np.ndarray,
    flanking: int,
) -> np.ndarray:
    """Mean log(mut/ref) using boundary windows for ref and central window for mut.

    Reference windows (1-based in df):
    - Left:  [left-1-(flanking-1) .. left-1]
    - Right: [right+1 .. right+flanking]
    Mutation windows (0-based center):
    - Left:  [center0-flanking .. center0-1]
    - Right: [center0 .. center0+flanking-1]
    Uses mutated base at the mutation index for channel selection.
    Returns per-row scores as a 1D numpy array.
    """
    base_idx = NUCLEOTIDE_TO_INDEX
    L = ref_probs.shape[1]
    center0 = L // 2
    mut_left0 = list(range(center0 - flanking, center0))
    mut_right0 = list(range(center0, center0 + flanking))

    scores = np.zeros(len(df), dtype=float)
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Scoring (SV effect)"):
        left1 = int(row["left"])   # 1-based
        right1 = int(row["right"]) # 1-based

        left_end = left1 - 1
        left_ref = list(range(left_end - (flanking - 1), left_end + 1))
        right_start = right1 + 1
        right_ref = list(range(right_start, right_start + flanking))

        vals = []
        mut_full = row["MutSeq"]
        mut_start0 = mut_left0[0]
        center_seq = mut_full[mut_start0 : mut_start0 + 2 * flanking]

        for k in range(flanking):
            # Left pair
            p_ref1 = left_ref[k]
            p_mut0 = mut_left0[k]
            b = center_seq[k].upper()
            if b in base_idx:
                j = base_idx[b]
                r = ref_probs[i, p_ref1 - 1, j]
                m = mut_probs[i, p_mut0, j]
                vals.append(float(np.log(max(m, 1e-12) / max(r, 1e-12))))
            else:
                vals.append(0.0)
            # Right pair
            p_ref1 = right_ref[k]
            p_mut0 = mut_right0[k]
            b = center_seq[flanking + k].upper()
            if b in base_idx:
                j = base_idx[b]
                r = ref_probs[i, p_ref1 - 1, j]
                m = mut_probs[i, p_mut0, j]
                vals.append(float(np.log(max(m, 1e-12) / max(r, 1e-12))))
            else:
                vals.append(0.0)
        scores[i] = float(np.mean(vals)) * -1
    return scores


def _compute_true_tokens_from_seq(sequences: pd.Series, positions: List[int]) -> np.ndarray:
    tokens = []
    for seq in sequences.tolist():
        for i in positions:
            tokens.append(seq[i].upper())
    return np.array(tokens)


def _metric_token_accuracy(probs: np.ndarray, true_tokens: np.ndarray) -> float:
    pred_idx = probs.argmax(axis=1)
    nuc = np.array(NUCLEOTIDES)
    pred = nuc[pred_idx]
    valid = np.isin(true_tokens, nuc)
    if not valid.any():
        return 0.0
    return float((pred[valid] == true_tokens[valid]).mean())


def _metric_motif_accuracy(probs: np.ndarray, true_tokens: np.ndarray, motif_len: int) -> float:
    nuc = np.array(NUCLEOTIDES)
    pred = nuc[probs.argmax(axis=1)]
    n = len(true_tokens)
    assert n % motif_len == 0, "total masked positions not divisible by motif_len"
    pred_groups = pred.reshape(-1, motif_len)
    true_groups = true_tokens.reshape(-1, motif_len)
    valid_groups = np.all(np.isin(true_groups, nuc), axis=1)
    if not valid_groups.any():
        return 0.0
    return float(np.all(pred_groups[valid_groups] == true_groups[valid_groups], axis=1).mean())


def _compute_auroc(df: pd.DataFrame, probs: np.ndarray, token_idx: int, seq_col: str) -> float:
    ref_series = df[seq_col].str[token_idx].str.upper()
    y_true = df["label"].astype(int)
    # Probability of REF base at masked index
    nuc = list(NUCLEOTIDES)
    ref_map = {b: i for i, b in enumerate(nuc)}
    scores = np.zeros(len(df), dtype=float)
    valid = ref_series.isin(nuc)
    idx = valid[valid].index
    scores[idx] = probs.reshape(len(df), -1)[valid.values, [ref_map[b] for b in ref_series[valid]]]
    fpr, tpr, _ = roc_curve(y_true, scores)
    return float(auc(fpr, tpr))

def _refprob_scores(df: pd.DataFrame, probs: np.ndarray, token_idx: int, seq_col: str) -> np.ndarray:
    ref_series = df[seq_col].str[token_idx].str.upper()
    nuc = list(NUCLEOTIDES)
    ref_map = {b: i for i, b in enumerate(nuc)}
    scores = np.zeros(len(df), dtype=float)
    valid = ref_series.isin(nuc)
    idx = valid[valid].index
    scores[idx] = probs.reshape(len(df), -1)[valid.values, [ref_map[b] for b in ref_series[valid]]]
    return scores


def _avg_trueprob_scores(
    probs: np.ndarray, true_tokens: np.ndarray, motif_len: int
) -> np.ndarray:
    """Average probability assigned to the ground-truth base at each masked position,
    grouped by motif_len. Returns one score per example.
    Unknown bases (not in A,C,G,T) are counted as 0.
    """
    nuc = np.array(NUCLEOTIDES)
    n = len(true_tokens)
    assert n % motif_len == 0, "total masked positions not divisible by motif_len"
    # Map each true token to index in A,C,G,T or -1 if unknown
    idx_map = {b: i for i, b in enumerate(nuc)}
    idxs = np.array([idx_map.get(t, -1) for t in true_tokens], dtype=int)
    # pick prob of true token, or 0 if idx == -1
    row_idx = np.arange(probs.shape[0])
    token_probs = np.zeros(probs.shape[0], dtype=float)
    valid = idxs >= 0
    token_probs[valid] = probs[row_idx[valid], idxs[valid]]
    # average per group
    return token_probs.reshape(-1, motif_len).mean(axis=1)


class ZeroShotEval:
    def evo_cons(
        self,
        repo_id: str = "",
        task: str = "",
        split: str = "valid",
        model: str = "kuleshov-group/PlantCAD2-Small-l24-d0768",
        token_idx: int = 255,
        batch_size: int = 128,
        seq_column: str = "sequence",
        save_logits: Optional[str] = None,
        logits_path: Optional[str] = None,
        metrics_json: Optional[str] = None,
        input_tsv: Optional[str] = None,
    ) -> None:
        """Compute masked-token probabilities at a single index and AUROC against labels.

        Expects dataset columns: `<seq_column>`, `label`. If `logits_path` is provided, loads TSV and skips inference.
        If `input_tsv` is provided, loads data from local TSV file instead of HuggingFace.
        """
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        accelerator = Accelerator()
        logger.info("Loading dataset")
        if input_tsv:
            logger.info(f"Loading data from local TSV: {input_tsv}")
            df = pd.read_csv(input_tsv, sep="\t")
        else:
            if not repo_id or not task:
                raise ValueError("Either input_tsv or both repo_id and task must be provided")
            ds = load_dataset(repo_id, task)
            df = ds[split].to_pandas()
        if logits_path is not None:
            probs = pd.read_csv(logits_path, sep="\t").values
        else:
            _require_cuda()
            model_, tok = _load_model(model)
            dataset = SingleMaskDataset(df[seq_column], tok, token_idx)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)
            model_, loader = accelerator.prepare(model_, loader)
            probs = _masked_probs(model_, tok, loader, accelerator, desc=f"Masked logits @ {token_idx}")
            if save_logits and accelerator.is_main_process:
                pd.DataFrame(probs, columns=list(NUCLEOTIDES)).to_csv(save_logits, sep="\t", index=False)
                logger.info(f"Saved logits TSV to {save_logits}")

        # Validate shape: one row per sequence
        assert probs.shape[0] == len(df), f"Row mismatch: probs={probs.shape[0]} examples={len(df)}"
        roc_auc = _compute_auroc(df, probs, token_idx, seq_column)
        # AUPRC using the same reference-base probability scores
        y_true = df["label"].astype(int).to_numpy()
        pr_scores = _refprob_scores(df, probs, token_idx, seq_column)
        auprc = float(average_precision_score(y_true, pr_scores))
        accelerator.print(f"AUROC\t{roc_auc:.6f}")
        accelerator.print(f"AUPRC\t{auprc:.6f}")
        if metrics_json and accelerator.is_main_process:
            with open(metrics_json, "w") as f:
                json.dump({"auroc": roc_auc, "auprc": auprc, "token_idx": token_idx}, f, indent=2)


    def motif_acc(
        self,
        repo_id: str = "",
        task: str = "",
        split: str = "valid",
        model: str = "kuleshov-group/PlantCAD2-Small-l24-d0768",
        mask_idx: Sequence[int] = (255, 256, 257),
        motif_len: int = 3,
        batch_size: int = 128,
        seq_column: str = "sequence",
        save_logits: Optional[str] = None,
        logits_path: Optional[str] = None,
        metrics_json: Optional[str] = None,
        input_tsv: Optional[str] = None,
    ) -> None:
        """Compute multi-position masked probabilities and token/motif accuracy.

        Expects dataset column: `<seq_column>`.
        If `logits_path` is provided, loads TSV of probabilities and skips model inference.
        If `input_tsv` is provided, loads data from local TSV file instead of HuggingFace.
        """
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        accelerator = Accelerator()
        logger.info("Loading dataset")
        if input_tsv:
            logger.info(f"Loading data from local TSV: {input_tsv}")
            df = pd.read_csv(input_tsv, sep="\t")
        else:
            if not repo_id or not task:
                raise ValueError("Either input_tsv or both repo_id and task must be provided")
            ds = load_dataset(repo_id, task)
            df = ds[split].to_pandas()

        # Fire parses sequences when annotated, so this accepts:
        # --mask-idx=1,2,3  or  --mask-idx="[1,2,3]"  etc.
        positions = [int(x) for x in mask_idx]
        assert len(positions) == motif_len, "mask_idx count must equal motif_len"

        if logits_path is not None:
            probs = pd.read_csv(logits_path, sep="\t").values
        else:
            _require_cuda()
            model_, tok = _load_model(model)
            dataset = MultiMaskDataset(df[seq_column], tok, positions)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)
            model_, loader = accelerator.prepare(model_, loader)
            probs = _masked_probs(model_, tok, loader, accelerator, desc=f"Masked logits motif_len={motif_len}")
            if save_logits and accelerator.is_main_process:
                pd.DataFrame(probs, columns=list(NUCLEOTIDES)).to_csv(save_logits, sep="\t", index=False)
                logger.info(f"Saved logits TSV to {save_logits}")

        expected = len(df) * len(positions)
        assert probs.shape[0] == expected, f"Row mismatch: probs={probs.shape[0]} expected={expected}"
        true_tokens = _compute_true_tokens_from_seq(df[seq_column], positions)
        token_acc = _metric_token_accuracy(probs, true_tokens)
        motif_acc = _metric_motif_accuracy(probs, true_tokens, motif_len)
        accelerator.print(f"token_accuracy\t{token_acc:.6f}")
        accelerator.print(f"motif_accuracy\t{motif_acc:.6f}")
        if metrics_json and accelerator.is_main_process:
            with open(metrics_json, "w") as f:
                json.dump({"token_accuracy": token_acc, "motif_accuracy": motif_acc}, f, indent=2)

    def sv_effect(
        self,
        repo_id: str = "",
        task: str = "",
        split: str = "valid",
        model: str = "kuleshov-group/PlantCAD2-Small-l24-d0768",
        batch_size: int = 64,
        flanking: int = 5,
        output: Optional[str] = None,
        save_ref_logits: Optional[str] = None,
        save_mut_logits: Optional[str] = None,
        input_tsv: Optional[str] = None,
    ) -> None:
        """Evaluate SV effect from a Hugging Face dataset split or local TSV file.

        Dataset must contain columns: RefSeq, MutSeq, left, right, label (labels as 0/1).
        Computes mean LLR score per row and prints AUPRC. Optionally writes scored TSV.
        If `input_tsv` is provided, loads data from local TSV file instead of HuggingFace.
        """
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        if input_tsv:
            logger.info(f"Loading data from local TSV: {input_tsv}")
            df = pd.read_csv(input_tsv, sep="\t")
        else:
            if not repo_id or not task:
                raise ValueError("Either input_tsv or both repo_id and task must be provided")
            ds = load_dataset(repo_id, task)
            df = ds[split].to_pandas()
        required = ["RefSeq", "MutSeq", "left", "right", "label"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"Missing required columns: {missing}")

        _require_cuda()
        accelerator = Accelerator()
        model_, tok = _load_model(model)

        ref_dataset = UnmaskedDataset(df["RefSeq"], tok)
        mut_dataset = UnmaskedDataset(df["MutSeq"], tok)
        ref_loader = DataLoader(ref_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
        mut_loader = DataLoader(mut_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
        model_, ref_loader, mut_loader = accelerator.prepare(model_, ref_loader, mut_loader)

        ref_probs = _unmasked_probs(ref_loader, model_, tok, accelerator, desc="Ref (unmasked)")
        mut_probs = _unmasked_probs(mut_loader, model_, tok, accelerator, desc="Mut (unmasked)")

        if save_ref_logits and accelerator.is_main_process:
            np.savez_compressed(save_ref_logits, logits=ref_probs)
        if save_mut_logits and accelerator.is_main_process:
            np.savez_compressed(save_mut_logits, logits=mut_probs)

        scores = _sv_llr_boundary(df, ref_probs, mut_probs, flanking)
        y_true = df["label"].astype(int).to_numpy()
        auprc = float(average_precision_score(y_true, scores))
        accelerator.print(f"AUPRC\t{auprc:.6f}")

        if output and accelerator.is_main_process:
            out_df = df.copy()
            out_df["score"] = scores
            out_df = out_df.drop(columns=["Left5_Positions", "Right5_Positions"], errors="ignore")
            out_df.to_csv(output, sep="\t", index=False)

    def core_noncore(
        self,
        repo_id: str = "",
        task: str = "",
        split: str = "valid",
        model: str = "kuleshov-group/PlantCAD2-Small-l24-d0768",
        mask_idx: Sequence[int] = (255, 256, 257),
        motif_len: int = 3,
        batch_size: int = 128,
        seq_column: str = "sequence",
        label_column: str = "label",
        save_logits: Optional[str] = None,
        logits_path: Optional[str] = None,
        metrics_json: Optional[str] = None,
        input_tsv: Optional[str] = None,
    ) -> None:
        """Core vs non-core classification via averaged true-base probability across masked positions.

        Expects dataset columns: `<seq_column>`, `<label_column>`.
        If `logits_path` is provided, loads TSV probabilities (A,C,G,T) and skips model inference.
        If `input_tsv` is provided, loads data from local TSV file instead of HuggingFace.
        Reports AUROC.
        """
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        accelerator = Accelerator()
        logger.info("Loading dataset")
        if input_tsv:
            logger.info(f"Loading data from local TSV: {input_tsv}")
            df = pd.read_csv(input_tsv, sep="\t")
        else:
            if not repo_id or not task:
                raise ValueError("Either input_tsv or both repo_id and task must be provided")
            ds = load_dataset(repo_id, task)
            df = ds[split].to_pandas()

        # Fire parses sequences when annotated, so this accepts:
        # --mask-idx=1,2,3  or  --mask-idx="[1,2,3]"  etc.
        positions = [int(x) for x in mask_idx]
        assert len(positions) == motif_len, "mask_idx count must equal motif_len"

        if logits_path is not None:
            probs = pd.read_csv(logits_path, sep="\t").values
        else:
            _require_cuda()
            model_, tok = _load_model(model)
            dataset = MultiMaskDataset(df[seq_column], tok, positions)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)
            model_, loader = accelerator.prepare(model_, loader)
            probs = _masked_probs(model_, tok, loader, accelerator, desc=f"Masked logits (core/non-core) motif_len={motif_len}")
            if save_logits and accelerator.is_main_process:
                pd.DataFrame(probs, columns=["A", "C", "G", "T"]).to_csv(save_logits, sep="\t", index=False)
                logger.info(f"Saved logits TSV to {save_logits}")

        expected = len(df) * len(positions)
        assert probs.shape[0] == expected, f"Row mismatch: probs={probs.shape[0]} expected={expected}"
        true_tokens = _compute_true_tokens_from_seq(df[seq_column], positions)
        scores = _avg_trueprob_scores(probs, true_tokens, motif_len)
        y_true = df[label_column].astype(int).to_numpy()
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = float(auc(fpr, tpr))
        auprc = float(average_precision_score(y_true, scores))
        accelerator.print(f"AUROC\t{roc_auc:.6f}")
        accelerator.print(f"AUPRC\t{auprc:.6f}")
        if metrics_json and accelerator.is_main_process:
            with open(metrics_json, "w") as f:
                json.dump({"auroc": roc_auc, "auprc": auprc}, f, indent=2)


def main():
    fire.Fire(ZeroShotEval)


if __name__ == "__main__":
    main()
