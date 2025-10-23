"""
Data collators for CodonTransformer2 masked language modeling.

This module provides collators that handle batch processing and masking strategies
for training codon sequence models.
"""

import json

import torch

# Mapping from codon token IDs to their corresponding amino acid mask token IDs
# This allows the model to learn codon usage by predicting specific codons from
# amino acid-level masked tokens.
TOKEN2MASK: dict[int, int] = {
    26: 13,  # K_AAA -> K*
    27: 16,  # N_AAC -> N*
    28: 13,  # K_AAG -> K*
    29: 16,  # N_AAT -> N*
    30: 21,  # T_ACA -> T*
    31: 21,  # T_ACC -> T*
    32: 21,  # T_ACG -> T*
    33: 21,  # T_ACT -> T*
    34: 19,  # R_AGA -> R*
    35: 20,  # S_AGC -> S*
    36: 19,  # R_AGG -> R*
    37: 20,  # S_AGT -> S*
    38: 12,  # I_ATA -> I*
    39: 12,  # I_ATC -> I*
    40: 15,  # M_ATG -> M*
    41: 12,  # I_ATT -> I*
    42: 18,  # Q_CAA -> Q*
    43: 11,  # H_CAC -> H*
    44: 18,  # Q_CAG -> Q*
    45: 11,  # H_CAT -> H*
    46: 17,  # P_CCA -> P*
    47: 17,  # P_CCC -> P*
    48: 17,  # P_CCG -> P*
    49: 17,  # P_CCT -> P*
    50: 19,  # R_CGA -> R*
    51: 19,  # R_CGC -> R*
    52: 19,  # R_CGG -> R*
    53: 19,  # R_CGT -> R*
    54: 14,  # L_CTA -> L*
    55: 14,  # L_CTC -> L*
    56: 14,  # L_CTG -> L*
    57: 14,  # L_CTT -> L*
    58: 8,  # E_GAA -> E*
    59: 7,  # D_GAC -> D*
    60: 8,  # E_GAG -> E*
    61: 7,  # D_GAT -> D*
    62: 5,  # A_GCA -> A*
    63: 5,  # A_GCC -> A*
    64: 5,  # A_GCG -> A*
    65: 5,  # A_GCT -> A*
    66: 10,  # G_GGA -> G*
    67: 10,  # G_GGC -> G*
    68: 10,  # G_GGG -> G*
    69: 10,  # G_GGT -> G*
    70: 22,  # V_GTA -> V*
    71: 22,  # V_GTC -> V*
    72: 22,  # V_GTG -> V*
    73: 22,  # V_GTT -> V*
    74: 25,  # __TAA -> _*
    75: 24,  # Y_TAC -> Y*
    76: 25,  # __TAG -> _*
    77: 24,  # Y_TAT -> Y*
    78: 20,  # S_TCA -> S*
    79: 20,  # S_TCC -> S*
    80: 20,  # S_TCG -> S*
    81: 20,  # S_TCT -> S*
    82: 25,  # __TGA -> _*
    83: 6,  # C_TGC -> C*
    84: 23,  # W_TGG -> W*
    85: 6,  # C_TGT -> C*
    86: 14,  # L_TTA -> L*
    87: 9,  # F_TTC -> F*
    88: 14,  # L_TTG -> L*
    89: 9,  # F_TTT -> F*
}


class MaskedTokenizerCollator:
    """
    Collator for masked language modeling on codon sequences.

    Implements a masking strategy for training codon prediction models:
    - 15% of tokens are selected for masking
    - Of selected tokens:
        - 80% are replaced with amino acid mask tokens (e.g., K_AAA -> K*)
        - 10% are replaced with random amino acid/codon tokens
        - 10% are kept unchanged

    This strategy enables the model to learn organism-specific codon usage patterns
    by predicting the original codon from masked amino acid representations.

    Args:
        tokenizer: HuggingFace tokenizer for codon sequences

    Attributes:
        tokenizer: The codon tokenizer instance
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
        # Leave special tokens and pads as they are.
        prob_matrix[inputs < 5] = 0.0
        selected = torch.bernoulli(prob_matrix).bool()

        # 80% of the time, selected input tokens are replaced with appropriate mask tokens.
        replaced = torch.bernoulli(torch.full(selected.shape, 0.8)).bool() & selected
        inputs[replaced] = self.token2mask_tensor[inputs[replaced]]

        # 10% of the time, selected tokens are replaced with a random codon (IDs 26-89).
        randomized = torch.bernoulli(torch.full(selected.shape, 0.5)).bool() & selected & ~replaced
        random_aa = torch.randint(26, 90, inputs.shape, dtype=torch.long)
        inputs[randomized] = random_aa[randomized]

        # Remaining 10% of selected tokens stay unchanged (handled implicitly)

        tokenized["input_ids"] = inputs
        tokenized["labels"] = torch.where(selected, targets, -100)

        # Convert species IDs to tensor
        species_ids = torch.tensor([int(s) for s in list_of_species], dtype=torch.long)
        tokenized["species_id"] = species_ids

        return tokenized
