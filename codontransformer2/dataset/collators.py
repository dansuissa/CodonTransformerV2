"""
Data collators for CodonTransformer2 masked language modeling.

This module provides collators that handle batch processing and masking strategies
for training codon sequence models.
"""

import json

import torch

from codontransformer2.dataset.constants import SYNONYMOUS_CODONS, TOKEN2MASK


class MaskedTokenizerCollator:
    """
    Collator for masked language modeling on codon sequences.

    Implements a masking strategy for training codon prediction models:
    - 15% of tokens are selected for masking
    - Of selected tokens:
        - 80% are replaced with amino acid mask tokens (e.g., K_AAA -> K*)
        - 10% are replaced with [MASK] token
        - 5% are replaced with random codon tokens from synonym list
        - 5% are kept unchanged

    This strategy enables the model to learn organism-specific codon usage patterns
    by predicting the original codon from masked amino acid representations.

    Args:
        tokenizer: HuggingFace tokenizer for codon sequences

    Attributes:
        tokenizer: The codon tokenizer instance
        mask_token_id: Token ID for [MASK] from tokenizer
        token2mask_tensor: Lookup tensor mapping codon IDs to amino acid mask IDs
    """

    def __init__(self, tokenizer):
        """
        Initialize the collator with a codon tokenizer.

        Args:
            tokenizer: PreTrainedTokenizerFast instance for codon sequences

        Returns:
            None
        """
        self.tokenizer = tokenizer
        self.mask_token_id = tokenizer.mask_token_id

        # Create a tensor lookup table for masking.
        self.token2mask_tensor = torch.tensor(
            [TOKEN2MASK.get(i, i) for i in range(90)], dtype=torch.long
        )

    def __call__(self, examples):
        """
        Process a batch of examples with masking for MLM training.

        Takes raw DNA sequences with species metadata, splits into codons,
        tokenizes, and applies masking strategy for training.

        Args:
            examples: List of dictionaries from WebDataset, each containing:
                - "json": JSON string with fields:
                    - "seq": DNA sequence string (e.g., "ATGGCATAG...")
                    - "species": Species/organism ID (numeric string or int)

        Returns:
            dict: Batch dictionary with tensors:
                - input_ids: Masked token IDs [batch_size, seq_len]
                - attention_mask: Attention mask [batch_size, seq_len]
                - labels: Target token IDs for loss computation [batch_size, seq_len]
                          (-100 for unmasked positions)
                - species_id: Species identifiers [batch_size]
        """
        list_of_species: list[str] = []
        list_of_codons: list[str] = []

        # Parse sequences and split into space-separated codons
        for ex in examples:
            doc = json.loads(ex["json"])
            seq = doc["seq"]
            # Use a generator expression to avoid creating new strings.
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

        inputs = tokenized["input_ids"]
        targets = inputs.clone()

        # Select 15% of tokens for masking (excluding special tokens)
        prob_matrix = torch.full(inputs.shape, 0.15)
        prob_matrix[inputs < 5] = 0.0  # Leave special tokens and pads as they are.
        selected = torch.bernoulli(prob_matrix).bool()

        # 80% of the time, selected input tokens are replaced with amino acid mask tokens.
        replaced = torch.bernoulli(torch.full(selected.shape, 0.8)).bool() & selected
        inputs[replaced] = self.token2mask_tensor[inputs[replaced]]

        # 10% of the time, selected tokens are replaced with [MASK] token.
        mask_token = torch.bernoulli(torch.full(selected.shape, 0.5)).bool() & selected & ~replaced
        inputs[mask_token] = self.mask_token_id

        # 5% of the time, replace with a random synonymous codon (same amino acid).
        random_synonym_mask = (
            torch.bernoulli(torch.full(selected.shape, 0.5)).bool()
            & selected
            & ~replaced
            & ~mask_token
        )
        for batch_index, token_index in random_synonym_mask.nonzero(as_tuple=False):
            original_token_id = int(targets[batch_index, token_index])
            synonym_candidates = SYNONYMOUS_CODONS.get(original_token_id, [original_token_id])
            sampled_index = torch.randint(0, len(synonym_candidates), (1,)).item()
            new_token_id = synonym_candidates[sampled_index]
            inputs[batch_index, token_index] = new_token_id

        # Remaining 5% of selected tokens stay unchanged (handled implicitly).
        tokenized["input_ids"] = inputs
        tokenized["labels"] = torch.where(selected, targets, -100)

        # Convert species IDs to tensor
        species_ids = torch.tensor([int(s) for s in list_of_species], dtype=torch.long)
        tokenized["species_id"] = species_ids

        return tokenized
