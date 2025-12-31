
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import torch
import torch.nn.functional as F

IGNORE_INDEX = -100

@torch.no_grad()
def masked_topk_accuracy(logits: torch.Tensor,labels: torch.Tensor,k: int = 1,ignore_index: int = IGNORE_INDEX,) -> torch.Tensor:
    mask = labels.ne(ignore_index)  # [B, L]
    denom = mask.sum().clamp_min(1) # [B, L, K]
    topk = logits.topk(k, dim=-1).indices # [B, L, 1]
    gold = labels.unsqueeze(-1)
    correct = (topk == gold).any(dim=-1) & mask  # [B, L]
    return correct.sum() / denom


@torch.no_grad()
def masked_accuracy(logits: torch.Tensor,labels: torch.Tensor,ignore_index: int = IGNORE_INDEX,) -> torch.Tensor:
    return masked_topk_accuracy(logits, labels, k=1, ignore_index=ignore_index)


@torch.no_grad()
def masked_loss_from_logits(logits: torch.Tensor,labels: torch.Tensor,ignore_index: int = IGNORE_INDEX,) -> torch.Tensor:
    B, L, V = logits.shape
    return F.cross_entropy(
        logits.view(B * L, V),
        labels.view(B * L),
        ignore_index=ignore_index,
        reduction="mean",
    )


#synonymous accuracy
@torch.no_grad()
def masked_synonymous_accuracy(logits: torch.Tensor,labels: torch.Tensor,codon_id_to_aa_id: torch.Tensor,k: int = 1,ignore_index: int = IGNORE_INDEX,) -> torch.Tensor:
    """
    Synonymous accuracy: predicted codon encodes the same amino acid as the gold codon.

    codon_id_to_aa_id: LongTensor [V] mapping vocab token id -> amino acid id
        (For non-codon/special tokens you can map to -1; those should be ignored anyway.)

    If k=1: checks top-1 codon synonymy.
    If k>1: checks if ANY of top-k predictions matches gold AA.
    """
    mask = labels.ne(ignore_index)
    denom = mask.sum().clamp_min(1)
    topk = logits.topk(k, dim=-1).indices  # [B, L, K]
    gold = labels  # [B, L]
    pred_aa = codon_id_to_aa_id[topk]                  # [B, L, K]
    gold_aa = codon_id_to_aa_id[gold].unsqueeze(-1)    # [B, L, 1]
    same = (pred_aa == gold_aa).any(dim=-1) & mask     # [B, L]
    return same.sum() / denom


# Species bucket support
@dataclass
class SpeciesBuckets:
    """
    Buckets are defined using trainingset frequency counts per species_id.
    - head: top `head_frac` most frequent species
    - tail: bottom `tail_frac` least frequent species
    - mid: the rest
    """
    head: torch.Tensor  # bool mask over species_id space: [S]
    mid: torch.Tensor   # [S]
    tail: torch.Tensor  # [S]


def build_species_buckets(species_counts: torch.Tensor,head_frac: float = 0.10,tail_frac: float = 0.50,) -> SpeciesBuckets:
    """
    species_counts: LongTensor [S] counts from TRAIN data for each species id.
    Returns boolean masks over species_id space of size S.
    Note: tail_frac=0.50 means "bottom 50% of species by frequency".
    """
    assert species_counts.dim() == 1
    S = species_counts.numel()
    sorted_ids = torch.argsort(species_counts, descending=True)  # [S]
    head_n = max(1, int(S * head_frac))
    tail_n = max(1, int(S * tail_frac))
    head_ids = sorted_ids[:head_n]
    tail_ids = sorted_ids[-tail_n:]
    mid_ids = sorted_ids[head_n:-tail_n] if head_n + tail_n < S else torch.tensor([], device=species_counts.device, dtype=torch.long)
    head = torch.zeros(S, dtype=torch.bool, device=species_counts.device)
    tail = torch.zeros(S, dtype=torch.bool, device=species_counts.device)
    mid  = torch.zeros(S, dtype=torch.bool, device=species_counts.device)
    head[head_ids] = True
    tail[tail_ids] = True
    if mid_ids.numel() > 0:
        mid[mid_ids] = True
    mid = ~(head | tail)
    return SpeciesBuckets(head=head, mid=mid, tail=tail)


@torch.no_grad()
def bucket_mask_from_species_ids(species_ids: torch.Tensor,bucket: torch.Tensor,) -> torch.Tensor:
    return bucket[species_ids]  # [B] bool


@torch.no_grad()
def forward_with_species_conditioning(model,species_embed: torch.nn.Embedding,batch: Dict[str, torch.Tensor],species_ids: torch.Tensor,) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      loss (scalar), logits [B,L,V]
    """
    input_ids = batch["input_ids"]
    attn = batch["attention_mask"]
    labels = batch["labels"]
    token_embeds = model.get_input_embeddings()(input_ids)     # [B,L,H]
    sp = species_embed(species_ids)                            # [B,H]
    token_embeds = token_embeds + sp[:, None, :]               # [B,L,H]
    out = model(inputs_embeds=token_embeds, attention_mask=attn, labels=labels)
    return out.loss, out.logits


@torch.no_grad()
def evaluate_mlm_batch(model,species_embed: torch.nn.Embedding,batch: Dict[str, torch.Tensor],*,topk: int = 5,codon_id_to_aa_id: Optional[torch.Tensor] = None,synonym_topk: int = 1,) -> Dict[str, torch.Tensor]:

    loss, logits = forward_with_species_conditioning(model, species_embed, batch, batch["species_id"])
    labels = batch["labels"]

    metrics = {
        "val_loss": loss.detach(),
        "val_acc": masked_accuracy(logits, labels),
        f"val_top{topk}_acc": masked_topk_accuracy(logits, labels, k=topk),
    }

    if codon_id_to_aa_id is not None:
        metrics["val_syn_acc"] = masked_synonymous_accuracy(logits, labels, codon_id_to_aa_id, k=synonym_topk)

    return metrics


@torch.no_grad()
def evaluate_species_ablations(model,species_embed: torch.nn.Embedding,batch: Dict[str, torch.Tensor],*,unknown_species_id: int,topk: int = 5,) -> Dict[str, torch.Tensor]:
    """
    Runs the same batch 3 times:
      - correct species
      - unknown species
      - wrong species (shuffled)
    """
    labels = batch["labels"]
    B = batch["species_id"].shape[0]
    device = batch["species_id"].device

    # 1) correct
    loss_c, logits_c = forward_with_species_conditioning(model, species_embed, batch, batch["species_id"])
    # 2) unknown
    unk_ids = torch.full((B,), unknown_species_id, device=device, dtype=torch.long)
    loss_u, logits_u = forward_with_species_conditioning(model, species_embed, batch, unk_ids)
    # 3) wrong (shuffle)
    perm = torch.randperm(B, device=device)
    wrong_ids = batch["species_id"][perm]
    loss_w, logits_w = forward_with_species_conditioning(model, species_embed, batch, wrong_ids)

    metrics = {
        "val_loss_correct": loss_c.detach(),
        "val_loss_unknown": loss_u.detach(),
        "val_loss_wrong": loss_w.detach(),
        "val_acc_correct": masked_accuracy(logits_c, labels),
        "val_acc_unknown": masked_accuracy(logits_u, labels),
        "val_acc_wrong": masked_accuracy(logits_w, labels),
        f"val_top{topk}_acc_correct": masked_topk_accuracy(logits_c, labels, k=topk),
        f"val_top{topk}_acc_unknown": masked_topk_accuracy(logits_u, labels, k=topk),
        f"val_top{topk}_acc_wrong": masked_topk_accuracy(logits_w, labels, k=topk),
    }

    # Shortcut indicators (loss deltas)
    metrics["delta_unk"] = (loss_u - loss_c).detach()
    metrics["delta_wrong"] = (loss_w - loss_c).detach()

    return metrics


@torch.no_grad()
def evaluate_tail_buckets(model,species_embed: torch.nn.Embedding,batch: Dict[str, torch.Tensor],buckets: SpeciesBuckets,*,topk: int = 5,) -> Dict[str, torch.Tensor]:
    """
    Computes loss/acc for head/mid/tail based on species_id frequency bucket.
    Note: For simplicity, loss is recomputed from logits with a masked CE.
    """
    loss, logits = forward_with_species_conditioning(model, species_embed, batch, batch["species_id"])
    labels = batch["labels"]
    species_ids = batch["species_id"]  # [B]
    device = labels.device
    metrics = {}

    def bucket_metrics(name: str, sample_mask: torch.Tensor):
        # sample_mask: [B] bool, selects samples in this bucket
        if sample_mask.sum().item() == 0:
            metrics[f"{name}_loss"] = torch.tensor(float("nan"), device=device)
            metrics[f"{name}_acc"] = torch.tensor(float("nan"), device=device)
            metrics[f"{name}_top{topk}_acc"] = torch.tensor(float("nan"), device=device)
            return

        logits_b = logits[sample_mask]   # [b,L,V]
        labels_b = labels[sample_mask]   # [b,L]

        metrics[f"{name}_loss"] = masked_loss_from_logits(logits_b, labels_b)
        metrics[f"{name}_acc"] = masked_accuracy(logits_b, labels_b)
        metrics[f"{name}_top{topk}_acc"] = masked_topk_accuracy(logits_b, labels_b, k=topk)

    bucket_metrics("val_head", buckets.head[species_ids])
    bucket_metrics("val_mid",  buckets.mid[species_ids])
    bucket_metrics("val_tail", buckets.tail[species_ids])

    return metrics
