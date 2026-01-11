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
    - mlm_probability (default 15%) of tokens are selected for masking
    - Of selected tokens:
        - 80% are replaced with amino acid mask tokens (e.g., K_AAA -> K*)
        - 10% are replaced with [MASK] token
        - 5% are replaced with random codon tokens from synonym list
        - 5% are kept unchanged

    This strategy enables the model to learn organism-specific codon usage patterns
    by predicting the original codon from masked amino acid representations.

    Key fixes vs previous version:
    - Supports both "seq" and "dna_sequence" JSON keys (your datasets use both).
    - Supports species name -> id mapping via species_to_id (instead of int(species)).
    - Supports unknown_species_id + max_species_id bounds checking.
    - Adds species_dropout_prob (replace species_id with unknown slot).
    - token2mask_tensor is sized to vocab_size (instead of hard-coded 90).
    - Excludes special tokens by explicit token IDs (instead of inputs < 5).
    - Optional trimming to a multiple of 3 to avoid partial codons at the end.

    Args:
        tokenizer: HuggingFace tokenizer for codon sequences
        species_to_id: Optional dict mapping species string -> int id
        unknown_species_id: Species id used for "unknown" / dropped-out species
        max_species_id: Optional upper bound; if out of range => unknown_species_id
        species_dropout_prob: Probability to replace species_id with unknown_species_id
        mlm_probability: Probability to select a token for masking (default 0.15)
        trim_to_multiple_of_3: If True, drops trailing bases so length is multiple of 3

    Attributes:
        tokenizer: The codon tokenizer instance
        mask_token_id: Token ID for [MASK] from tokenizer
        token2mask_tensor: Lookup tensor mapping codon IDs to amino acid mask IDs
    """

    def __init__(
        self,
        tokenizer,
        *,
        species_to_id=None,
        unknown_species_id: int = 0,
        max_species_id: int | None = None,
        species_dropout_prob: float = 0.0,
        mlm_probability: float = 0.15,
        trim_to_multiple_of_3: bool = True,
    ):
        """
        Initialize the collator with a codon tokenizer.

        Args:
            tokenizer: PreTrainedTokenizerFast instance for codon sequences
            species_to_id: Optional mapping from species string to integer id.
            unknown_species_id: id to use when species is missing/unmapped/out-of-range.
            max_species_id: if set, enforces 0 <= species_id < max_species_id.
            species_dropout_prob: probability to replace species_id with unknown_species_id.
            mlm_probability: fraction of (non-special) tokens selected for masking.
            trim_to_multiple_of_3: remove trailing bases so we only form full codons.

        Returns:
            None
        """
        self.tokenizer = tokenizer
        self.mask_token_id = tokenizer.mask_token_id

        # --- Species handling (NEW) ---
        self.species_to_id = species_to_id
        self.unknown_species_id = int(unknown_species_id)
        self.max_species_id = max_species_id
        self.species_dropout_prob = float(species_dropout_prob)

        # --- Masking hyperparam (NEW) ---
        self.mlm_probability = float(mlm_probability)

        # --- Sequence hygiene (NEW) ---
        self.trim_to_multiple_of_3 = bool(trim_to_multiple_of_3)

        # Create a tensor lookup table for masking.
        # FIX: Previously this was hard-coded to 90 and could crash if vocab_size != 90.
        vocab_size = len(tokenizer)
        self.token2mask_tensor = torch.arange(vocab_size, dtype=torch.long)
        for codon_id, aa_mask_id in TOKEN2MASK.items():
            codon_id = int(codon_id)
            if 0 <= codon_id < vocab_size:
                self.token2mask_tensor[codon_id] = int(aa_mask_id)

        # FIX: Previously special tokens were excluded via `inputs < 5`, which is fragile.
        # Now we exclude by explicit token IDs.
        self._special_ids = set()
        for tid in (
            tokenizer.pad_token_id,
            tokenizer.cls_token_id,
            tokenizer.sep_token_id,
            tokenizer.unk_token_id,
        ):
            if tid is not None:
                self._special_ids.add(int(tid))

    def _to_species_id(self, raw_species):
        """
        Convert raw species field to an integer id.

        Accepts:
        - int
        - numeric string
        - species name string (if species_to_id is provided)

        If missing/unmapped/out-of-range => unknown_species_id.
        """
        sid = self.unknown_species_id

        if raw_species is None:
            sid = self.unknown_species_id
        elif isinstance(raw_species, int):
            sid = raw_species
        else:
            s = str(raw_species)
            if self.species_to_id is not None:
                sid = self.species_to_id.get(s, self.unknown_species_id)
            else:
                try:
                    sid = int(s)
                except ValueError:
                    sid = self.unknown_species_id

        # Bounds check if requested
        if self.max_species_id is not None:
            if not (0 <= int(sid) < int(self.max_species_id)):
                sid = self.unknown_species_id

        return int(sid)

    def __call__(self, examples):
        """
        Process a batch of examples with masking for MLM training.

        Takes raw DNA sequences with species metadata, splits into codons,
        tokenizes, and applies masking strategy for training.

        Args:
            examples: List of dictionaries from WebDataset, each containing:
                - "json": JSON string/bytes with fields:
                    - "seq" or "dna_sequence": DNA sequence string (e.g., "ATGGCATAG...")
                    - "species": species identifier; can be numeric id OR species name
                               (name -> id requires species_to_id)

        Returns:
            dict: Batch dictionary with tensors:
                - input_ids: Masked token IDs [batch_size, seq_len]
                - attention_mask: Attention mask [batch_size, seq_len]
                - labels: Target token IDs for loss computation [batch_size, seq_len]
                          (-100 for unmasked positions)
                - species_id: Species identifiers [batch_size]
        """
        list_of_species: list[int] = []
        list_of_codons: list[str] = []

        # Parse sequences and split into space-separated codons
        for ex in examples:
            doc = json.loads(ex["json"])

            # FIX: Support both keys ("seq" in shards, "dna_sequence" in per-organism JSONs)
            seq = doc.get("seq", None)
            if seq is None:
                seq = doc.get("dna_sequence", None)
            if seq is None:
                continue

            # FIX: Optional trimming to avoid a trailing incomplete codon (helps reduce [UNK])
            if self.trim_to_multiple_of_3:
                seq = seq[: (len(seq) // 3) * 3]
                if len(seq) == 0:
                    continue

            # Use a generator expression to avoid creating new strings.
            codons = " ".join(seq[i : i + 3] for i in range(0, len(seq), 3))

            # FIX: Species can be name string -> map; also supports unknown/bounds-check
            list_of_species.append(self._to_species_id(doc.get("species", None)))
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

        # Select mlm_probability of tokens for masking (excluding special tokens)
        prob_matrix = torch.full(inputs.shape, self.mlm_probability)

        # FIX: Exclude special tokens by explicit ids (pad/cls/sep/unk), not by `inputs < 5`.
        special_mask = torch.zeros_like(inputs, dtype=torch.bool)
        for tid in self._special_ids:
            special_mask |= (inputs == tid)
        prob_matrix[special_mask] = 0.0

        selected = torch.bernoulli(prob_matrix).bool()

        # 80% of the time, selected input tokens are replaced with amino acid mask tokens.
        replaced = torch.bernoulli(torch.full(selected.shape, 0.8)).bool() & selected
        inputs[replaced] = self.token2mask_tensor[inputs[replaced]]

        # 10% of the time, selected tokens are replaced with [MASK] token.
        # (half of the remaining 20% of selected tokens)
        mask_token = torch.bernoulli(torch.full(selected.shape, 0.5)).bool() & selected & ~replaced
        inputs[mask_token] = self.mask_token_id

        # 5% of the time, replace with a random synonymous codon (same amino acid).
        # (half of the remaining 10% of selected tokens)
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
        species_ids = torch.tensor(list_of_species, dtype=torch.long)

        # FIX: Species dropout (NEW) replace some species_ids with unknown_species_id
        if self.species_dropout_prob > 0.0:
            drop = torch.rand(species_ids.shape) < self.species_dropout_prob
            species_ids = torch.where(
                drop,
                torch.full_like(species_ids, self.unknown_species_id),
                species_ids,
            )

        tokenized["species_id"] = species_ids

        return tokenized
