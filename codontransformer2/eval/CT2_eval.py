from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import torch
import torch.nn.functional as F
IGNORE_INDEX = -100

@torch.no_grad()
def masked_topk_accuracy(logits: torch.Tensor,labels: torch.Tensor,k: int = 1,ignore_index: int = IGNORE_INDEX,) -> torch.Tensor:

    mask = labels.ne(ignore_index)  # [B, L]
    denom = mask.sum().clamp_min(1)
    topk = logits.topk(k, dim=-1).indices  # [B, L, K]
    gold = labels.unsqueeze(-1)  # [B, L, 1]
    correct = (topk == gold).any(dim=-1) & mask  # [B, L]
    return correct.sum() / denom


@torch.no_grad()
def masked_accuracy(logits: torch.Tensor,labels: torch.Tensor,ignore_index: int = IGNORE_INDEX,) -> torch.Tensor:
    return masked_topk_accuracy(logits, labels, k=1, ignore_index=ignore_index)


@torch.no_grad()
def masked_loss_from_logits(logits: torch.Tensor,labels: torch.Tensor,ignore_index: int = IGNORE_INDEX,reduction: str = "mean",) -> torch.Tensor:

    B, L, V = logits.shape
    return F.cross_entropy(
        logits.view(B * L, V),
        labels.view(B * L),
        ignore_index=ignore_index,
        reduction=reduction,)



# Synonymous accuracy
@torch.no_grad()
def masked_synonymous_accuracy(logits: torch.Tensor,labels: torch.Tensor,codon_id_to_aa_id: torch.Tensor,k: int = 1,ignore_index: int = IGNORE_INDEX,) -> torch.Tensor:

    mask = labels.ne(ignore_index)  # [B, L]
    denom = mask.sum().clamp_min(1)
    topk = logits.topk(k, dim=-1).indices  # [B, L, K]
    gold = labels  # [B, L]
    pred_aa = codon_id_to_aa_id[topk]  # [B, L, K]
    gold_aa = codon_id_to_aa_id[gold].unsqueeze(-1)  # [B, L, 1]
    same = (pred_aa == gold_aa).any(dim=-1) & mask  # [B, L]
    return same.sum() / denom

# Species bucket support
@dataclass
class SpeciesBuckets:
    """
    Buckets defined using training frequency counts per species_id.
    head: top head_frac most frequent species
    tail: bottom tail_frac least frequent species
    mid: everything else
    """
    head: torch.Tensor  
    mid: torch.Tensor   # [S]
    tail: torch.Tensor  # [S]


@torch.no_grad()
def build_species_buckets(species_counts: torch.Tensor,head_frac: float = 0.10,tail_frac: float = 0.50,) -> SpeciesBuckets:
    if (not torch.is_tensor(species_counts)) or species_counts.dim() != 1:
        raise ValueError(f"species_counts must be a 1D tensor, got {type(species_counts)} shape={getattr(species_counts, 'shape', None)}")

    S = species_counts.numel()
    sorted_ids = torch.argsort(species_counts, descending=True)  # [S]
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
def bucket_mask_from_species_ids(species_ids: torch.Tensor, bucket: torch.Tensor) -> torch.Tensor:
    return bucket[species_ids]


@torch.no_grad()
def forward_with_species_conditioning(model,species_embed: torch.nn.Embedding,batch: Dict[str, torch.Tensor],species_ids: torch.Tensor,) -> Tuple[torch.Tensor, torch.Tensor]:
    
    input_ids = batch["input_ids"]
    attn = batch["attention_mask"]
    labels = batch["labels"]
    token_embeds = model.get_input_embeddings()(input_ids)  # [B,L,H]
    sp = species_embed(species_ids)                         # [B,H]
    token_embeds = token_embeds + sp[:, None, :]            # [B,L,H]
    out = model(inputs_embeds=token_embeds, attention_mask=attn, labels=labels)
    return out.loss, out.logits

@torch.no_grad()
def evaluate_mlm_batch(model,species_embed: torch.nn.Embedding,batch: Dict[str, torch.Tensor],*,topk: int = 5,codon_id_to_aa_id: Optional[torch.Tensor] = None,synonym_topk: int = 1,prefix: str = "val_",) -> Dict[str, torch.Tensor]:

    loss, logits = forward_with_species_conditioning(model, species_embed, batch, batch["species_id"])
    labels = batch["labels"]

    metrics: Dict[str, torch.Tensor] = {
        f"{prefix}loss": loss.detach(),
        f"{prefix}acc": masked_accuracy(logits, labels),
        f"{prefix}top{topk}_acc": masked_topk_accuracy(logits, labels, k=topk),
    }

    if codon_id_to_aa_id is not None:
        metrics[f"{prefix}syn_acc"] = masked_synonymous_accuracy(
            logits, labels, codon_id_to_aa_id, k=synonym_topk
        )

    return metrics


@torch.no_grad()
def evaluate_species_ablations( model, species_embed: torch.nn.Embedding, batch: Dict[str, torch.Tensor], *, unknown_species_id: int, topk: int = 5, prefix: str = "val_",) -> Dict[str, torch.Tensor]:

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

    metrics: Dict[str, torch.Tensor] = {
        f"{prefix}loss_correct": loss_c.detach(),
        f"{prefix}loss_unknown": loss_u.detach(),
        f"{prefix}loss_wrong": loss_w.detach(),
        f"{prefix}acc_correct": masked_accuracy(logits_c, labels),
        f"{prefix}acc_unknown": masked_accuracy(logits_u, labels),
        f"{prefix}acc_wrong": masked_accuracy(logits_w, labels),
        f"{prefix}top{topk}_acc_correct": masked_topk_accuracy(logits_c, labels, k=topk),
        f"{prefix}top{topk}_acc_unknown": masked_topk_accuracy(logits_u, labels, k=topk),
        f"{prefix}top{topk}_acc_wrong": masked_topk_accuracy(logits_w, labels, k=topk),
        f"{prefix}delta_unk": (loss_u - loss_c).detach(),
        f"{prefix}delta_wrong": (loss_w - loss_c).detach(),
    }
    return metrics


@torch.no_grad()
def evaluate_tail_buckets(model,species_embed: torch.nn.Embedding,batch: Dict[str, torch.Tensor],buckets: SpeciesBuckets,*,topk: int = 5,prefix: str = "val_",) -> Dict[str, torch.Tensor]:
    """
    Computes loss/acc for head/mid/tail based on species frequency buckets.
    Loss is recomputed from logits with masked CE (so it can be computed on sub-batches).
"""
    _, logits = forward_with_species_conditioning(model, species_embed, batch, batch["species_id"])
    labels = batch["labels"]
    species_ids = batch["species_id"]  # [B]
    device = labels.device

    metrics: Dict[str, torch.Tensor] = {}

    def bucket_metrics(name: str, sample_mask: torch.Tensor):
        if sample_mask.sum().item() == 0:
            metrics[f"{prefix}{name}_loss"] = torch.tensor(float("nan"), device=device)
            metrics[f"{prefix}{name}_acc"] = torch.tensor(float("nan"), device=device)
            metrics[f"{prefix}{name}_top{topk}_acc"] = torch.tensor(float("nan"), device=device)
            return

        logits_b = logits[sample_mask]  # [b,L,V]
        labels_b = labels[sample_mask]  # [b,L]

        metrics[f"{prefix}{name}_loss"] = masked_loss_from_logits(logits_b, labels_b)
        metrics[f"{prefix}{name}_acc"] = masked_accuracy(logits_b, labels_b)
        metrics[f"{prefix}{name}_top{topk}_acc"] = masked_topk_accuracy(logits_b, labels_b, k=topk)

    bucket_metrics("head", buckets.head[species_ids])
    bucket_metrics("mid", buckets.mid[species_ids])
    bucket_metrics("tail", buckets.tail[species_ids])

    return metrics



class MacroMicroAccumulator:

    def __init__(self, ignore_index: int = IGNORE_INDEX):
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.micro_correct = 0.0
        self.micro_total = 0.0
        self.micro_loss_sum = 0.0 
        self.per_species = {}  

    @torch.no_grad()
    def update(self, logits: torch.Tensor, labels: torch.Tensor, *, species_ids: Optional[torch.Tensor] = None):
        device = logits.device
        mask = labels.ne(self.ignore_index)  # [B,L]
        total = mask.sum().item()
        if total <= 0:
            return
        B, L, V = logits.shape
        loss_sum = F.cross_entropy(
            logits.view(B * L, V),
            labels.view(B * L),
            ignore_index=self.ignore_index,
            reduction="sum",
        ).item()

        pred = logits.argmax(dim=-1)  # [B,L]
        correct = ((pred == labels) & mask).sum().item()
        self.micro_correct += correct
        self.micro_total += total
        self.micro_loss_sum += loss_sum
        if species_ids is None:
            return
        # per-species aggregation
        # i aggregate per sample (sequence) but count masked tokens within each sample
        # so species macro is effectively "avg across species of token-level metrics".
        
        for i in range(labels.shape[0]):
            sid = int(species_ids[i].item())
            m_i = mask[i]
            tot_i = m_i.sum().item()
            if tot_i == 0:
                continue
            # loss sum and correct for this sample
            logits_i = logits[i:i+1]  # [1,L,V]
            labels_i = labels[i:i+1]  # [1,L]
            loss_i = F.cross_entropy(
                logits_i.view(L, V),
                labels_i.view(L),
                ignore_index=self.ignore_index,
                reduction="sum",
            ).item()
            pred_i = logits_i.argmax(dim=-1).view(-1)  # [L]
            corr_i = ((pred_i == labels_i.view(-1)) & m_i.view(-1)).sum().item()

            rec = self.per_species.get(sid)
            if rec is None:
                rec = {"loss_sum": 0.0, "correct": 0.0, "total": 0.0}
                self.per_species[sid] = rec
            rec["loss_sum"] += loss_i
            rec["correct"] += corr_i
            rec["total"] += tot_i

    def compute_micro(self) -> Dict[str, float]:
        denom = max(1.0, self.micro_total)
        return {
            "loss_micro": self.micro_loss_sum / denom,
            "acc_micro": self.micro_correct / denom,
            "masked_tokens": self.micro_total,
        }

    def compute_macro(self) -> Dict[str, float]:
        if len(self.per_species) == 0:
            return {"loss_macro": float("nan"), "acc_macro": float("nan"), "n_species": 0}

        losses = []
        accs = []
        for sid, rec in self.per_species.items():
            tot = rec["total"]
            if tot <= 0:
                continue
            losses.append(rec["loss_sum"] / tot)
            accs.append(rec["correct"] / tot)

        if len(losses) == 0:
            return {"loss_macro": float("nan"), "acc_macro": float("nan"), "n_species": len(self.per_species)}

        return {
            "loss_macro": float(sum(losses) / len(losses)),
            "acc_macro": float(sum(accs) / len(accs)),
            "n_species": len(self.per_species),
        }
