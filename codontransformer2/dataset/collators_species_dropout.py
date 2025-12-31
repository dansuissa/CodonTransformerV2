
from __future__ import annotations
import json
from typing import Dict, List, Optional
import torch
from codontransformer2.dataset.constants import SYNONYMOUS_CODONS, TOKEN2MASK
class MaskedTokenizerCollator:

    def __init__(
        self,
        tokenizer,
        *,
        species_to_id: Optional[Dict[str, int]] = None,
        unknown_species_id: int = 0,
        max_species_id: Optional[int] = None,
        species_dropout_prob: float = 0.0,
        mlm_probability: float = 0.15,
    ):
        self.tokenizer = tokenizer
        self.mask_token_id = tokenizer.mask_token_id
        self.pad_token_id = tokenizer.pad_token_id

        self.species_to_id = species_to_id
        self.unknown_species_id = int(unknown_species_id)
        self.max_species_id = int(max_species_id) if max_species_id is not None else None
        self.species_dropout_prob = float(species_dropout_prob)
        self.mlm_probability = float(mlm_probability)

        # Build codon->aa-mask lookup tensor for the *whole vocab* (safe if vocab > 90).
        vocab_size = len(tokenizer)
        lut = [TOKEN2MASK.get(i, i) for i in range(vocab_size)]
        self.token2mask_tensor = torch.tensor(lut, dtype=torch.long)

        # Special ids (exclude PAD because we handle it too)
        self.special_ids = list(getattr(tokenizer, "all_special_ids", []))
        if self.pad_token_id in self.special_ids:
            self.special_ids.remove(self.pad_token_id)

    def _parse_json(self, x) -> Dict:
        if isinstance(x, (bytes, bytearray)):
            x = x.decode("utf-8", errors="replace")
        return json.loads(x)

    def _get_species_id(self, species_value) -> int:
        # prefer mapping by name
        if self.species_to_id is not None:
            sid = self.species_to_id.get(str(species_value), None)
            if sid is None:
                return self.unknown_species_id
            sid = int(sid)
        else:
            # fallback: if numeric string, use int; else unknown
            try:
                sid = int(species_value)
            except Exception:
                sid = self.unknown_species_id

        if self.max_species_id is not None:
            if not (0 <= sid < self.max_species_id):
                sid = self.unknown_species_id
        return sid

    def __call__(self, examples: List[Dict]):
        # Build raw codon strings + species ids
        codon_texts: List[str] = []
        species_ids: List[int] = []

        for ex in examples:
            doc = self._parse_json(ex["json"])

            # DNA sequence field name(s)
            seq = (
                doc.get("dna_sequence")
                or doc.get("seq")
                or doc.get("dna")
                or doc.get("sequence")
            )
            if seq is None:
                raise KeyError("JSON sample missing dna sequence field (expected 'dna_sequence' or 'seq').")

            species = doc.get("species", None)
            sid = self._get_species_id(species)
            species_ids.append(sid)

            # Convert DNA -> space-separated codons
            # (keeps last partial codon if exists; tokenizer truncation handles it anyway)
            codons = " ".join(seq[i : i + 3] for i in range(0, len(seq), 3))
            codon_texts.append(codons)

        # Tokenize
        tokenized = self.tokenizer(
            codon_texts,
            return_attention_mask=True,
            return_token_type_ids=False,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        inputs = tokenized["input_ids"]
        labels = inputs.clone()

        # Select tokens for masking (exclude PAD and all special tokens)
        selected = torch.full(inputs.shape, self.mlm_probability, dtype=torch.float)
        special_mask = inputs.eq(self.pad_token_id)
        for sid in self.special_ids:
            special_mask |= inputs.eq(sid)
        selected[special_mask] = 0.0
        selected = torch.bernoulli(selected).bool()

        # 80% -> AA-mask token
        replaced = torch.bernoulli(torch.full(inputs.shape, 0.8)).bool() & selected
        inputs[replaced] = self.token2mask_tensor[inputs[replaced]]

        # 10% -> [MASK] (half of remaining 20%)
        mask_token = (
            torch.bernoulli(torch.full(inputs.shape, 0.5)).bool()
            & selected
            & ~replaced
        )
        inputs[mask_token] = self.mask_token_id

        # 5% -> random synonymous codon (half of remaining 10%)
        random_synonym_mask = (
            torch.bernoulli(torch.full(inputs.shape, 0.5)).bool()
            & selected
            & ~replaced
            & ~mask_token
        )

        # Note: loop is OK because only ~0.75% tokens overall hit this branch (15% * 5%).
        nz = random_synonym_mask.nonzero(as_tuple=False)
        for b_idx, t_idx in nz:
            orig_id = int(labels[b_idx, t_idx])
            syns = SYNONYMOUS_CODONS.get(orig_id, [orig_id])
            new_id = syns[int(torch.randint(0, len(syns), (1,)).item())]
            inputs[b_idx, t_idx] = new_id

        # Labels: only masked positions contribute
        tokenized["input_ids"] = inputs
        tokenized["labels"] = torch.where(selected, labels, torch.full_like(labels, -100))

        # Species tensor (+ optional species dropout)
        sp = torch.tensor(species_ids, dtype=torch.long)
        if self.species_dropout_prob > 0.0:
            drop = torch.rand(sp.shape[0]) < self.species_dropout_prob
            sp[drop] = int(self.unknown_species_id)

        tokenized["species_id"] = sp
        return tokenized
