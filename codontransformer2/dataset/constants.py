"""
Constants for the dataset.
"""

# Mapping from codon token IDs to their corresponding amino acid mask token IDs
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


SYNONYMOUS_CODONS: dict[int, list[int]] = {
    26: [26, 28],  # K_AAA -> K_AAA, K_AAG
    27: [27, 29],  # N_AAC -> N_AAC, N_AAT
    28: [26, 28],  # K_AAG -> K_AAA, K_AAG
    29: [27, 29],  # N_AAT -> N_AAT, N_AAC
    30: [30, 31, 32, 33],  # T_ACA -> T_ACA, T_ACC, T_ACG, T_ACT
    31: [30, 31, 32, 33],  # T_ACA -> T_ACA, T_ACC, T_ACG, T_ACT
    32: [30, 31, 32, 33],  # T_ACA -> T_ACA, T_ACC, T_ACG, T_ACT
    33: [30, 31, 32, 33],  # T_ACA -> T_ACA, T_ACC, T_ACG, T_ACT
    34: [34, 36, 50, 51, 52, 53],  # R_CGA -> R_CGA, R_CGC, R_CGG, R_CGT
    35: [35, 37, 78, 79, 80, 81],  # S_AGC -> S_AGC, S_AGT, S_TCA, S_TCC, S_TCG, S_TCT
    36: [34, 36, 50, 51, 52, 53],  # R_CGA -> R_CGA, R_CGC, R_CGG, R_CGT
    37: [35, 37, 78, 79, 80, 81],  # S_AGC -> S_AGC, S_AGT, S_TCA, S_TCC, S_TCG, S_TCT
    38: [38, 39, 41],  # I_ATA -> I_ATA, I_ATC
    39: [38, 39, 41],  # I_ATA -> I_ATA, I_ATC
    40: [40],  # M_ATG -> M_ATG
    41: [38, 39, 41],  # I_ATA -> I_ATA, I_ATC
    42: [42, 44],  # Q_CAA -> Q_CAA, Q_CAG
    43: [43, 45],  # H_CAC -> H_CAC, H_CAT
    44: [42, 44],  # Q_CAA -> Q_CAA, Q_CAG
    45: [43, 45],  # H_CAC -> H_CAC, H_CAT
    46: [46, 47, 48, 49],  # P_CCA -> P_CCA, P_CCC, P_CCG, P_CCT
    47: [46, 47, 48, 49],  # P_CCA -> P_CCA, P_CCC, P_CCG, P_CCT
    48: [46, 47, 48, 49],  # P_CCA -> P_CCA, P_CCC, P_CCG, P_CCT
    49: [46, 47, 48, 49],  # P_CCA -> P_CCA, P_CCC, P_CCG, P_CCT
    50: [34, 36, 50, 51, 52, 53],  # R_CGA -> R_CGA, R_CGC, R_CGG, R_CGT
    51: [34, 36, 50, 51, 52, 53],  # R_CGA -> R_CGA, R_CGC, R_CGG, R_CGT
    52: [34, 36, 50, 51, 52, 53],  # R_CGA -> R_CGA, R_CGC, R_CGG, R_CGT
    53: [34, 36, 50, 51, 52, 53],  # R_CGA -> R_CGA, R_CGC, R_CGG, R_CGT
    54: [54, 55, 56, 57, 86, 88],  # L_CTA -> L_CTA, L_CTC, L_CTG, L_CTT, L_TTG, L_TTT
    55: [54, 55, 56, 57, 86, 88],  # L_CTA -> L_CTA, L_CTC, L_CTG, L_CTT, L_TTG, L_TTT
    56: [54, 55, 56, 57, 86, 88],  # L_CTA -> L_CTA, L_CTC, L_CTG, L_CTT, L_TTG, L_TTT
    57: [54, 55, 56, 57, 86, 88],  # L_CTA -> L_CTA, L_CTC, L_CTG, L_CTT, L_TTG, L_TTT
    58: [58, 60],  # E_GAA -> E_GAA, E_GAG
    59: [59, 61],  # D_GAC -> D_GAC, D_GAT
    60: [58, 60],  # E_GAA -> E_GAA, E_GAG
    61: [59, 61],  # D_GAC -> D_GAC, D_GAT
    62: [62, 63, 64, 65],  # A_GCA -> A_GCA, A_GCC, A_GCG, A_GCT
    63: [62, 63, 64, 65],  # A_GCA -> A_GCA, A_GCC, A_GCG, A_GCT
    64: [62, 63, 64, 65],  # A_GCA -> A_GCA, A_GCC, A_GCG, A_GCT
    65: [62, 63, 64, 65],  # A_GCA -> A_GCA, A_GCC, A_GCG, A_GCT
    66: [66, 67, 68, 69],  # G_GGA -> G_GGA, G_GGC, G_GGG, G_GGT
    67: [66, 67, 68, 69],  # G_GGA -> G_GGA, G_GGC, G_GGG, G_GGT
    68: [66, 67, 68, 69],  # G_GGA -> G_GGA, G_GGC, G_GGG, G_GGT
    69: [66, 67, 68, 69],  # G_GGA -> G_GGA, G_GGC, G_GGG, G_GGT
    70: [70, 71, 72, 73],  # V_GTA -> V_GTA, V_GTC, V_GTG, V_GTT
    71: [70, 71, 72, 73],  # V_GTA -> V_GTA, V_GTC, V_GTG, V_GTT
    72: [70, 71, 72, 73],  # V_GTA -> V_GTA, V_GTC, V_GTG, V_GTT
    73: [70, 71, 72, 73],  # V_GTA -> V_GTA, V_GTC, V_GTG, V_GTT
    74: [74, 76, 82],  # __TAA -> __TAA, __TAG
    75: [75, 77],  # Y_TAC -> Y_TAC, Y_TAT
    76: [74, 76, 82],  # __TAA -> __TAA, __TAG
    77: [75, 77],  # Y_TAC -> Y_TAC, Y_TAT
    78: [35, 37, 78, 79, 80, 81],  # S_TCA -> S_TCA, S_TCG, S_TCT
    79: [35, 37, 78, 79, 80, 81],  # S_TCA -> S_TCA, S_TCG, S_TCT
    80: [35, 37, 78, 79, 80, 81],  # S_TCA -> S_TCA, S_TCG, S_TCT
    81: [35, 37, 78, 79, 80, 81],  # S_TCA -> S_TCA, S_TCG, S_TCT
    82: [74, 76, 82],  # __TAA -> __TAA, __TAG
    83: [83, 85],  # C_TGC -> C_TGC, C_TGT
    84: [84],  # W_TGG -> W_TGG
    85: [83, 85],  # C_TGC -> C_TGC, C_TGT
    86: [54, 55, 56, 57, 86, 88],  # L_CTA -> L_CTA, L_CTC, L_CTG, L_CTT, L_TTG, L_TTT
    87: [87, 89],  # F_TTC -> F_TTC, F_TTT
    88: [54, 55, 56, 57, 86, 88],  # L_CTA -> L_CTA, L_CTC, L_CTG, L_CTT, L_TTG, L_TTT
    89: [87, 89],  # F_TTC -> F_TTC, F_TTT
}
