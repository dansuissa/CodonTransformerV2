"""
Data collators for CodonTransformer2 masked language modeling.

This module provides collators that handle batch processing and masking strategies
for training codon sequence models.

This version adds optional species dropout:
- with probability `species_drop_p`, a sample's species_id is replaced by `unknown_species_id`.

Notes:
- This collator assumes each WebDataset sample has a "json" field containing a JSON string.
"""

import json
from typing import Any, Dict, List, Optional

import torch

from codontransformer2.dataset.constants import SYNONYMOUS_CODONS, TOKEN2MASK


class MaskedTokenizerCollator:
    """
    Collator for masked language modeling on codon sequences.

    Masking strategy (for selected 15% of tokens; excluding special tokens/pads):
        - 80% replaced with amino acid mask tokens (e.g., K_AAA -> K*)
        - 10% replaced with [MASK]
        - 5% replaced with a random synonymous codon (same amino acid)
        - 5% kept unchanged

    Optional species dropout:
        - with probability `species_drop_p`, replace species_id with `unknown_species_id`

    Args:
        tokenizer: HuggingFace tokenizer for codon sequences
        species_drop_p: Probability of dropping/replacing the species_id per sample (default: 0.0)
        unknown_species_id: The integer species id to use when dropping species (required if species_drop_p > 0)
        special_token_id_threshold: Keep tokens with id < threshold unmasked (default: 5).
            This assumes special tokens/pad are assigned small IDs. If your tokenizer differs, consider
            switching to tokenizer.get_special_tokens_mask.
        mask_prob: Fraction of tokens selected for MLM corruption (default: 0.15)

    Returns (per batch):
        - input_ids: [B, L]
        - attention_mask: [B, L]
        - labels: [B, L] (with -100 for non-selected positions)
        - species_id: [B]
    """

    def __init__(
        self,
        tokenizer,
        species_drop_p: float = 0.0,
        unknown_species_id: Optional[int] = None,
        special_token_id_threshold: int = 5,
        mask_prob: float = 0.15,
    ):
        self.tokenizer = tokenizer
        self.mask_token_id = tokenizer.mask_token_id

        if not (0.0 <= species_drop_p <= 1.0):
            raise ValueError(f"species_drop_p must be in [0,1], got {species_drop_p}")
        self.species_drop_p = float(species_drop_p)
        self.unknown_species_id = unknown_species_id
        if self.species_drop_p > 0.0 and self.unknown_species_id is None:
            raise ValueError("unknown_species_id must be provided when species_drop_p > 0")

        if not (0.0 <= mask_prob <= 1.0):
            raise ValueError(f"mask_prob must be in [0,1], got {mask_prob}")
        self.mask_prob = float(mask_prob)

        self.special_token_id_threshold = int(special_token_id_threshold)

        # Create a tensor lookup table for masking codon IDs to amino-acid-only mask IDs.
        # The original code assumes vocab size around 90 for this mapping.
        self.token2mask_tensor = torch.tensor(
            [TOKEN2MASK.get(i, i) for i in range(90)], dtype=torch.long
        )

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        list_of_species: List[str] = []
        list_of_codons: List[str] = []

        # Parse sequences and split into space-separated codons
        for ex in examples:
            doc = json.loads(ex["json"])

            # Original expected field name is "seq". If your dataset uses "dna_sequence",
            # change this line accordingly:
            seq = doc["seq"]

            # Split into codons (assumes len(seq) is multiple of 3; if not, last chunk is shorter)
            codons = " ".join(seq[i : i + 3] for i in range(0, len(seq), 3))

            list_of_species.append(doc["species"])
            list_of_codons.append(codons)

        # Tokenize codon sequences
        tokenized = self.tokenizer(
            list_of_codons,
            return_attention_mask=True,
            return_token_type_ids=False,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        inputs: torch.Tensor = tokenized["input_ids"]              # [B, L]
        targets: torch.Tensor = inputs.clone()                     # [B, L]

        # Select mask_prob of tokens for masking (excluding special tokens/pads)
        prob_matrix = torch.full(inputs.shape, self.mask_prob, dtype=torch.float)
        prob_matrix[inputs < self.special_token_id_threshold] = 0.0
        selected = torch.bernoulli(prob_matrix).bool()            # [B, L]

        # 80% of selected -> AA mask token
        replaced = torch.bernoulli(torch.full(selected.shape, 0.8)).bool() & selected
        inputs[replaced] = self.token2mask_tensor[inputs[replaced]]

        # From remaining selected (20%): 50% -> [MASK]  => overall 10% of selected
        mask_token = (
            torch.bernoulli(torch.full(selected.shape, 0.5)).bool()
            & selected
            & ~replaced
        )
        inputs[mask_token] = self.mask_token_id

        # From remaining selected (10%): 50% -> random synonym => overall 5% of selected
        random_synonym_mask = (
            torch.bernoulli(torch.full(selected.shape, 0.5)).bool()
            & selected
            & ~replaced
            & ~mask_token
        )

        # Replace with a random synonymous codon (same amino acid). This loop is OK but can be slower
        # for very large batches; optimize later if needed.
        for batch_index, token_index in random_synonym_mask.nonzero(as_tuple=False):
            original_token_id = int(targets[batch_index, token_index])
            synonym_candidates = SYNONYMOUS_CODONS.get(original_token_id, [original_token_id])
            sampled_index = torch.randint(0, len(synonym_candidates), (1,)).item()
            inputs[batch_index, token_index] = synonym_candidates[sampled_index]

        # Remaining selected positions stay unchanged (overall 5% of selected)
        tokenized["input_ids"] = inputs
        tokenized["labels"] = torch.where(selected, targets, torch.full_like(targets, -100))

        # Convert species IDs to tensor
        # If your dataset stores species as strings (names), you must map them to ints before this.
        species_ids = torch.tensor([int(s) for s in list_of_species], dtype=torch.long)

        # Optional species dropout
        if self.species_drop_p > 0.0:
            drop_mask = torch.rand(species_ids.shape) < self.species_drop_p
            species_ids = torch.where(
                drop_mask,
                torch.full_like(species_ids, int(self.unknown_species_id)),
                species_ids,
            )

        tokenized["species_id"] = species_ids
        return tokenized
