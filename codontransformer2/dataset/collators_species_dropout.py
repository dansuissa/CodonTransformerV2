from __future__ import annotations

import json
import torch
from codontransformer2.dataset.constants import SYNONYMOUS_CODONS, TOKEN2MASK


class MaskedTokenizerCollator:
    """
    Final CT2 collator.

    Input JSON expected:
      {
        "species": "...",
        "dna_sequence": "...",
        "protein_sequence": "..."
      }

    Converts:
      DNA + protein -> lowercase aa_codon tokens

    Example:
      ATG + M -> m_atg
      GAT + D -> d_gat
      TAA + * -> __taa

    Supports:
      - MLM masking
      - amino-acid mask replacement via TOKEN2MASK
      - random synonymous codon replacement
      - species_id mapping
      - species dropout
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
        full_mask_probability: float = 0.0,
        trim_to_multiple_of_3: bool = True,
        max_length: int | None = None,
    ):
        self.tokenizer = tokenizer

        # Ensure special tokens are registered, even if tokenizer JSON only has them in vocab.
        self.tokenizer.add_special_tokens({
            "unk_token": "[UNK]",
            "cls_token": "[CLS]",
            "sep_token": "[SEP]",
            "pad_token": "[PAD]",
            "mask_token": "[MASK]",
        })

        self.mask_token_id = self.tokenizer.mask_token_id
        self.pad_token_id = self.tokenizer.pad_token_id

        if self.mask_token_id is None:
            raise ValueError("tokenizer.mask_token_id is None.")
        if self.pad_token_id is None:
            raise ValueError("tokenizer.pad_token_id is None.")

        self.species_to_id = species_to_id
        self.unknown_species_id = int(unknown_species_id)
        self.max_species_id = int(max_species_id) if max_species_id is not None else None
        self.species_dropout_prob = float(species_dropout_prob)
        self.mlm_probability = float(mlm_probability)
        self.full_mask_probability = float(full_mask_probability)
        self.trim_to_multiple_of_3 = bool(trim_to_multiple_of_3)
        self.max_length = max_length

        vocab_size = len(self.tokenizer)
        self.token2mask_tensor = torch.arange(vocab_size, dtype=torch.long)
        for codon_id, aa_mask_id in TOKEN2MASK.items():
            codon_id = int(codon_id)
            aa_mask_id = int(aa_mask_id)
            if 0 <= codon_id < vocab_size:
                self.token2mask_tensor[codon_id] = aa_mask_id

        self.special_ids = set()
        for tid in (
            self.tokenizer.pad_token_id,
            self.tokenizer.cls_token_id,
            self.tokenizer.sep_token_id,
            self.tokenizer.unk_token_id,
        ):
            if tid is not None:
                self.special_ids.add(int(tid))

        self.vocab = self.tokenizer.get_vocab()

    def _parse_json(self, raw):
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", errors="replace")
        return json.loads(raw)

    def _to_species_id(self, raw_species) -> int:
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

        sid = int(sid)

        if self.max_species_id is not None and not (0 <= sid < self.max_species_id):
            sid = self.unknown_species_id

        return sid

    def _build_token_text(self, dna: str, protein: str | None) -> str:
        dna = str(dna).upper().replace("U", "T")

        if self.trim_to_multiple_of_3:
            dna = dna[: (len(dna) // 3) * 3]

        if len(dna) == 0:
            return ""

        toks = []

        if protein is not None:
            protein = str(protein)
            n_codons = min(len(dna) // 3, len(protein))

            for i in range(n_codons):
                codon = dna[3 * i : 3 * i + 3].lower()
                aa = protein[i].lower()

                if aa in ("*", "_"):
                    token = f"__{codon}"
                else:
                    token = f"{aa}_{codon}"

                if token in self.vocab:
                    toks.append(token)
        else:
            # Fallback only. This will likely produce [UNK] unless tokenizer has raw codons.
            for i in range(0, len(dna), 3):
                token = dna[i : i + 3].lower()
                if token in self.vocab:
                    toks.append(token)

        return " ".join(toks)

    def __call__(self, examples):
        texts = []
        species_ids = []

        for ex in examples:
            doc = self._parse_json(ex["json"])

            dna = (
                doc.get("dna_sequence")
                or doc.get("seq")
                or doc.get("dna")
                or doc.get("sequence")
            )
            if dna is None:
                continue

            protein = doc.get("protein_sequence")
            text = self._build_token_text(dna, protein)

            if not text:
                continue

            texts.append(text)
            species_ids.append(self._to_species_id(doc.get("species")))

        if len(texts) == 0:
            return {
                "input_ids": torch.empty((0, 0), dtype=torch.long),
                "attention_mask": torch.empty((0, 0), dtype=torch.long),
                "labels": torch.empty((0, 0), dtype=torch.long),
                "species_id": torch.empty((0,), dtype=torch.long),
            }

        tokenized = self.tokenizer(
            texts,
            return_attention_mask=True,
            return_token_type_ids=False,
            truncation=self.max_length is not None,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )

        inputs = tokenized["input_ids"]
        labels = inputs.clone()

        prob = torch.full(inputs.shape, self.mlm_probability, dtype=torch.float)

        special_mask = torch.zeros_like(inputs, dtype=torch.bool)
        for tid in self.special_ids:
            special_mask |= inputs.eq(tid)
        prob[special_mask] = 0.0

        selected = torch.bernoulli(prob).bool()

        # Stage 5 option:
        # With probability full_mask_probability, convert an entire sequence into
        # generation-style input: amino-acid mask tokens everywhere, labels everywhere.
        if self.full_mask_probability > 0.0:
            full_rows = torch.bernoulli(
                torch.full((inputs.shape[0], 1), self.full_mask_probability, dtype=torch.float)
            ).bool()
            full_selected = full_rows & ~special_mask
        else:
            full_selected = torch.zeros_like(inputs, dtype=torch.bool)

        selected = selected | full_selected

        # Full-mask rows should use amino-acid-specific mask tokens, not [MASK]/random.
        inputs[full_selected] = self.token2mask_tensor[inputs[full_selected]]

        normal_selected = selected & ~full_selected

        # 80% of normal MLM selections -> amino-acid mask token
        replaced = torch.bernoulli(torch.full(inputs.shape, 0.8)).bool() & normal_selected
        inputs[replaced] = self.token2mask_tensor[inputs[replaced]]

        # 10% of normal MLM selections -> [MASK]
        mask_token = (
            torch.bernoulli(torch.full(inputs.shape, 0.5)).bool()
            & normal_selected
            & ~replaced
        )
        inputs[mask_token] = self.mask_token_id

        # 5% of normal MLM selections -> random synonymous codon
        random_synonym_mask = (
            torch.bernoulli(torch.full(inputs.shape, 0.5)).bool()
            & normal_selected
            & ~replaced
            & ~mask_token
        )

        for b_idx, t_idx in random_synonym_mask.nonzero(as_tuple=False):
            original_id = int(labels[b_idx, t_idx])
            candidates = SYNONYMOUS_CODONS.get(original_id, [original_id])
            new_id = candidates[int(torch.randint(0, len(candidates), (1,)).item())]
            inputs[b_idx, t_idx] = new_id

        tokenized["input_ids"] = inputs
        tokenized["labels"] = torch.where(
            selected,
            labels,
            torch.full_like(labels, -100),
        )

        sp = torch.tensor(species_ids, dtype=torch.long)
        if self.species_dropout_prob > 0.0:
            drop = torch.rand(sp.shape) < self.species_dropout_prob
            sp = torch.where(
                drop,
                torch.full_like(sp, self.unknown_species_id),
                sp,
            )

        tokenized["species_id"] = sp
        return tokenized
