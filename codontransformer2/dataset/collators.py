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

    Masking strategy:
    - mlm_probability of (non-special) tokens are selected
    - Of selected tokens:
        - 80% replaced with amino-acid mask tokens via TOKEN2MASK lookup (e.g., k_aaa -> k*)
        - 10% replaced with [MASK]
        - 5% replaced with random synonymous codon token (same amino acid)
        - 5% kept unchanged

    IMPORTANT: This collator builds input tokens as "aa_codon" (e.g., "m_atg"),
    using BOTH protein_sequence + dna_sequence when available. This matches the
    lowercase CodonTransformerTokenizer you have (vocab uses m_atg, k_aaa, __taa, ...).

    Args:
        tokenizer: HuggingFace tokenizer for codon sequences (must have mask_token_id)
        species_to_id: Optional dict mapping species string -> int id
        unknown_species_id: Species id used for "unknown" / dropped-out species
        max_species_id: Optional upper bound; if out of range => unknown_species_id
        species_dropout_prob: Probability to replace species_id with unknown_species_id
        mlm_probability: Probability to select a token for masking (default 0.15)
        trim_to_multiple_of_3: If True, drops trailing bases so length is multiple of 3
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
        self.tokenizer = tokenizer
        self.mask_token_id = tokenizer.mask_token_id
        if self.mask_token_id is None:
            raise ValueError(
                "tokenizer.mask_token_id is None. Configure/add a [MASK] token on the tokenizer before using this collator."
            )

        # Species handling
        self.species_to_id = species_to_id
        self.unknown_species_id = int(unknown_species_id)
        self.max_species_id = max_species_id
        self.species_dropout_prob = float(species_dropout_prob)

        # Masking hyperparam
        self.mlm_probability = float(mlm_probability)

        # Sequence hygiene
        self.trim_to_multiple_of_3 = bool(trim_to_multiple_of_3)

        # token2mask lookup sized to full vocab
        vocab_size = len(tokenizer)
        self.token2mask_tensor = torch.arange(vocab_size, dtype=torch.long)
        for codon_id, aa_mask_id in TOKEN2MASK.items():
            codon_id = int(codon_id)
            if 0 <= codon_id < vocab_size:
                self.token2mask_tensor[codon_id] = int(aa_mask_id)

        # Exclude special tokens by explicit token IDs
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

        if self.max_species_id is not None:
            if not (0 <= int(sid) < int(self.max_species_id)):
                sid = self.unknown_species_id

        return int(sid)

    def __call__(self, examples):
        list_of_species: list[int] = []
        list_of_texts: list[str] = []

        for ex in examples:
            raw = ex["json"]
            if isinstance(raw, (bytes, bytearray)):
                raw = raw.decode("utf-8", errors="replace")
            doc = json.loads(raw)

            dna = doc.get("dna_sequence") or doc.get("seq")
            prot = doc.get("protein_sequence")

            if dna is None:
                continue

            # Optional trimming (safe even if already divisible by 3)
            if self.trim_to_multiple_of_3:
                dna = dna[: (len(dna) // 3) * 3]
                if len(dna) == 0:
                    continue

            # Build aa_codon tokens if protein_sequence exists
            if prot is not None:
                n_codons = min(len(dna) // 3, len(prot))
                if n_codons == 0:
                    continue

                toks = []
                for i in range(n_codons):
                    codon = dna[3 * i : 3 * i + 3].lower()
                    aa = prot[i].lower()

                    # Stop handling (rare in your shards, but safe)
                    if aa in ("*", "_"):
                        toks.append(f"__{codon}")     # "__taa"
                    else:
                        toks.append(f"{aa}_{codon}")  # "m_atg"

                text = " ".join(toks)
            else:
                # Fallback if protein_sequence is missing
                text = " ".join(dna[i : i + 3].lower() for i in range(0, len(dna), 3))

            list_of_species.append(self._to_species_id(doc.get("species", None)))
            list_of_texts.append(text)

        if len(list_of_texts) == 0:
            # DataLoader will interpret an empty dict as "skip"
            # but returning an empty batch is usually less confusing.
            return {
                "input_ids": torch.empty((0, 0), dtype=torch.long),
                "attention_mask": torch.empty((0, 0), dtype=torch.long),
                "labels": torch.empty((0, 0), dtype=torch.long),
                "species_id": torch.empty((0,), dtype=torch.long),
            }

        # Tokenize
        tokenized = self.tokenizer(
            list_of_texts,
            return_attention_mask=True,
            return_token_type_ids=False,
            truncation=False,  # avoids "no max length" warning unless you set max_length
            padding=True,
            return_tensors="pt",
        )

        inputs = tokenized["input_ids"]
        targets = inputs.clone()

        # Select mlm_probability of tokens for masking (excluding special tokens)
        prob_matrix = torch.full(inputs.shape, self.mlm_probability)

        special_mask = torch.zeros_like(inputs, dtype=torch.bool)
        for tid in self._special_ids:
            special_mask |= (inputs == tid)
        prob_matrix[special_mask] = 0.0

        selected = torch.bernoulli(prob_matrix).bool()

        # 80% -> amino-acid mask token via lookup
        replaced = torch.bernoulli(torch.full(selected.shape, 0.8)).bool() & selected
        inputs[replaced] = self.token2mask_tensor[inputs[replaced]]

        # 10% -> [MASK] token (half of remaining 20%)
        mask_token = torch.bernoulli(torch.full(selected.shape, 0.5)).bool() & selected & ~replaced
        inputs[mask_token] = self.mask_token_id

        # 5% -> random synonymous codon (half of remaining 10%)
        random_synonym_mask = (
            torch.bernoulli(torch.full(selected.shape, 0.5)).bool()
            & selected
            & ~replaced
            & ~mask_token
        )
        for batch_index, token_index in random_synonym_mask.nonzero(as_tuple=False):
            original_token_id = int(targets[batch_index, token_index])
            synonym_candidates = SYNONYMOUS_CODONS.get(original_token_id, [original_token_id])
            new_token_id = synonym_candidates[torch.randint(0, len(synonym_candidates), (1,)).item()]
            inputs[batch_index, token_index] = new_token_id

        # Labels: only masked positions contribute
        tokenized["input_ids"] = inputs
        tokenized["labels"] = torch.where(selected, targets, -100)

        # Species tensor (+ optional dropout)
        species_ids = torch.tensor(list_of_species, dtype=torch.long)
        if self.species_dropout_prob > 0.0:
            drop = torch.rand(species_ids.shape) < self.species_dropout_prob
            species_ids = torch.where(
                drop,
                torch.full_like(species_ids, self.unknown_species_id),
                species_ids,
            )
        tokenized["species_id"] = species_ids

        return tokenized
