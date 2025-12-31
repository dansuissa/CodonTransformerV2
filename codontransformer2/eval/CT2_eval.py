from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


@torch.no_grad()
def masked_topk_accuracy(logits: torch.Tensor,labels: torch.Tensor,k: int = 1,ignore_index: int = IGNORE_INDEX,) -> torch.Tensor:
    mask = labels.ne(ignore_index)  # [B,L]
    denom = mask.sum().clamp_min(1).to(logits.dtype)
    topk = logits.topk(k, dim=-1).indices  # [B,L,K]
    gold = labels.unsqueeze(-1)            # [B,L,1]
    correct = (topk == gold).any(dim=-1) & mask  # [B,L]
    return correct.sum().to(logits.dtype) / denom


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


@torch.no_grad()
def masked_synonymous_accuracy(logits: torch.Tensor,labels: torch.Tensor,codon_id_to_aa_id: torch.Tensor,  # [V] -> aa_id (or -1)k: int = 1,ignore_index: int = IGNORE_INDEX,) -> torch.Tensor:

    mask = labels.ne(ignore_index)  # [B,L]
    denom = mask.sum().clamp_min(1).to(logits.dtype)
    topk = logits.topk(k, dim=-1).indices  # [B,L,K]
    gold = labels                            # [B,L]
    pred_aa = codon_id_to_aa_id[topk]                 # [B,L,K]
    gold_aa = codon_id_to_aa_id[gold].unsqueeze(-1)   # [B,L,1]
    valid_gold = gold_aa.ne(-1).squeeze(-1)  # [B,L]
    mask = mask & valid_gold
    same = (pred_aa == gold_aa).any(dim=-1) & mask  # [B,L]
    return same.sum().to(logits.dtype) / denom


@dataclass
class SpeciesBuckets:
    head: torch.Tensor  # [S] bool
    mid: torch.Tensor   # [S] bool
    tail: torch.Tensor  # [S] bool


def build_species_buckets(species_counts: torch.Tensor,head_frac: float = 0.10, tail_frac: float = 0.50,) -> SpeciesBuckets:
    assert species_counts.dim() == 1
    S = species_counts.numel()
    sorted_ids = torch.argsort(species_counts, descending=True)
    head_n = max(1, int(S * head_frac))
    tail_n = max(1, int(S * tail_frac))
    head_ids = sorted_ids[:head_n]
    tail_ids = sorted_ids[-tail_n:]
    head = torch.zeros(S, dtype=torch.bool, device=species_counts.device)
    tail = torch.zeros(S, dtype=torch.bool, device=species_counts.device)
    head[head_ids] = True
    tail[tail_ids] = True
    mid = ~(head | tail)
    return SpeciesBuckets(head=head, mid=mid, tail=tail)


@torch.no_grad()
def forward_with_species_conditioning(model,species_embed: torch.nn.Embedding,batch: Dict[str, torch.Tensor],species_ids: torch.Tensor, ) -> Tuple[torch.Tensor, torch.Tensor]:
    input_ids = batch["input_ids"]
    attn = batch["attention_mask"]
    labels = batch["labels"]
    token_embeds = model.get_input_embeddings()(input_ids)  # [B,L,H]
    sp = species_embed(species_ids)                         # [B,H]
    token_embeds = token_embeds + sp[:, None, :]            # [B,L,H]
    out = model(inputs_embeds=token_embeds, attention_mask=attn, labels=labels)
    return out.loss, out.logits


@torch.no_grad()
def evaluate_mlm_batch(model,species_embed: torch.nn.Embedding,batch: Dict[str, torch.Tensor],*,topk: int = 5,codon_id_to_aa_id: Optional[torch.Tensor] = None,synonym_topk: int = 1,prefix: str = "test",) -> Dict[str, torch.Tensor]:
    loss, logits = forward_with_species_conditioning(model, species_embed, batch, batch["species_id"])
    labels = batch["labels"]
    metrics: Dict[str, torch.Tensor] = {
        f"{prefix}_loss": loss.detach(),
        f"{prefix}_acc": masked_accuracy(logits, labels),
        f"{prefix}_top{topk}_acc": masked_topk_accuracy(logits, labels, k=topk),
    }
    if codon_id_to_aa_id is not None:
        metrics[f"{prefix}_syn_acc"] = masked_synonymous_accuracy(
            logits, labels, codon_id_to_aa_id, k=synonym_topk
        )
    return metrics


@torch.no_grad()
def evaluate_species_ablations(model,species_embed: torch.nn.Embedding,batch: Dict[str, torch.Tensor],*,unknown_species_id: int,topk: int = 5,prefix: str = "test",) -> Dict[str, torch.Tensor]:
    labels = batch["labels"]
    B = batch["species_id"].shape[0]
    device = batch["species_id"].device
    # correct
    loss_c, logits_c = forward_with_species_conditioning(model, species_embed, batch, batch["species_id"])
    # unknown
    unk_ids = torch.full((B,), int(unknown_species_id), device=device, dtype=torch.long)
    loss_u, logits_u = forward_with_species_conditioning(model, species_embed, batch, unk_ids)
    # wrong (shuffle)
    perm = torch.randperm(B, device=device)
    wrong_ids = batch["species_id"][perm]
    loss_w, logits_w = forward_with_species_conditioning(model, species_embed, batch, wrong_ids)

    metrics = {
        f"{prefix}_loss_correct": loss_c.detach(),
        f"{prefix}_loss_unknown": loss_u.detach(),
        f"{prefix}_loss_wrong": loss_w.detach(),
        f"{prefix}_acc_correct": masked_accuracy(logits_c, labels),
        f"{prefix}_acc_unknown": masked_accuracy(logits_u, labels),
        f"{prefix}_acc_wrong": masked_accuracy(logits_w, labels),
        f"{prefix}_top{topk}_acc_correct": masked_topk_accuracy(logits_c, labels, k=topk),
        f"{prefix}_top{topk}_acc_unknown": masked_topk_accuracy(logits_u, labels, k=topk),
        f"{prefix}_top{topk}_acc_wrong": masked_topk_accuracy(logits_w, labels, k=topk),
        f"{prefix}_delta_unk": (loss_u - loss_c).detach(),
        f"{prefix}_delta_wrong": (loss_w - loss_c).detach(),
    }
    return metrics


@torch.no_grad()
def evaluate_tail_buckets(model,species_embed: torch.nn.Embedding,batch: Dict[str, torch.Tensor],buckets: SpeciesBuckets,*,topk: int = 5,prefix: str = "test",) -> Dict[str, torch.Tensor]:
    loss, logits = forward_with_species_conditioning(model, species_embed, batch, batch["species_id"])
    labels = batch["labels"]
    species_ids = batch["species_id"]  # [B]
    device = labels.device
    metrics: Dict[str, torch.Tensor] = {}

    def bucket_metrics(name: str, sample_mask: torch.Tensor):
        if sample_mask.sum().item() == 0:
            metrics[f"{prefix}_{name}_loss"] = torch.tensor(float("nan"), device=device)
            metrics[f"{prefix}_{name}_acc"] = torch.tensor(float("nan"), device=device)
            metrics[f"{prefix}_{name}_top{topk}_acc"] = torch.tensor(float("nan"), device=device)
            return

        logits_b = logits[sample_mask]
        labels_b = labels[sample_mask]
        metrics[f"{prefix}_{name}_loss"] = masked_loss_from_logits(logits_b, labels_b)
        metrics[f"{prefix}_{name}_acc"] = masked_accuracy(logits_b, labels_b)
        metrics[f"{prefix}_{name}_top{topk}_acc"] = masked_topk_accuracy(logits_b, labels_b, k=topk)

    bucket_metrics("head", buckets.head[species_ids])
    bucket_metrics("mid", buckets.mid[species_ids])
    bucket_metrics("tail", buckets.tail[species_ids])

    return metrics
