# backup of the court names
court_names_backup = ['AG_AK', 'AG_LWRK', 'AG_OG', 'AG_OGA', 'AG_OSB', 'AG_PRG', 'AG_RGAR', 'AG_SKBG', 'AG_SRG', 'AG_SVWG', 'AG_VB', 'AG_XX', 'AI_BZG', 'AI_KG', 'AI_XX', 'AR_KG', 'AR_OG', 'AR_RR', 'AR_SRK', 'AR_VB', 'AR_XX', 'BE_AK', 'BE_NAB', 'BE_OG', 'BE_SRK', 'BE_VB', 'BE_VG', 'BE_XX', 'BL_EG', 'BL_KG', 'BL_SG', 'BL_XX', 'BL_ZMG', 'BS_APG', 'BS_SVG', 'BS_XX', 'CH_BGE', 'CH_BGer', 'CH_BSTG', 'CH_BVGE', 'CH_EDÖB', 'CH_PATG', 'CH_VB', 'CH_WBK', 'CH_XX', 'FR_TAMA', 'FR_TC', 'FR_XX', 'GE_CAPJ', 'GE_CJ', 'GE_TAPI', 'GE_TP', 'GE_XX', 'GL_KG', 'GL_OG', 'GL_VG', 'GL_XX', 'GR_KG', 'GR_VG', 'GR_XX', 'JU_TC', 'JU_TPI', 'JU_XX', 'LU_AUK', 'LU_BKD', 'LU_BZG', 'LU_GSD', 'LU_JSD', 'LU_KG', 'LU_RR', 'LU_RSH', 'LU_XX', 'NE_ARA', 'NE_ASL', 'NE_ASS', 'NE_ATS', 'NE_CA', 'NE_CN', 'NE_TC', 'NE_TR', 'NE_XX', 'NW_OG', 'NW_XX', 'OW_OG', 'OW_VB', 'OW_VG', 'OW_XX', 'SG_ABSK', 'SG_HG', 'SG_KG', 'SG_KGN', 'SG_VB', 'SG_VG', 'SG_VGN', 'SG_VSG', 'SG_VWEK', 'SG_XX', 'SH_OG', 'SH_XX', 'SO_OG', 'SO_SK', 'SO_STG', 'SO_VG', 'SO_VSG', 'SO_XX', 'SZ_KG', 'SZ_XX', 'TG_OG', 'TG_XX', 'TI_CARP', 'TI_CATI', 'TI_CRP', 'TI_GIAR', 'TI_GPC', 'TI_PP', 'TI_TCA', 'TI_TCAS', 'TI_TE', 'TI_TPC', 'TI_TRAC', 'TI_TRAP', 'TI_TRPI', 'TI_XX', 'UR_REB', 'UR_XX', 'VD_SR', 'VD_TC', 'VD_TN', 'VD_TPHA', 'VD_XX', 'VS_AG', 'VS_BZG', 'VS_SRK', 'VS_TC', 'VS_XX', 'ZG_VG', 'ZG_XX', 'ZH_BK', 'ZH_BRK', 'ZH_HG', 'ZH_KSG', 'ZH_OG', 'ZH_SOBE', 'ZH_SRK', 'ZH_SVG', 'ZH_VG', 'ZH_XX', 'ZG_UPL', 'BE_UPL', 'SG_OG', 'FR_UPL', 'VD_UPL', 'ZH_UPL', 'AG_JG', 'AG_VG', 'AG_HG', 'AG_VSG', 'LU_UPL', 'CH_UPL', 'AG_UPL', 'AG_AUK', 'AG_RR', 'AG_JL', 'AI_UPL', 'AR_UPL', 'BL_UPL', 'BS_UPL', 'GE_UPL', 'GL_UPL', 'GR_UPL', 'JU_UPL', 'NE_UPL', 'NW_UPL', 'OW_UPL', 'SG_UPL', 'SH_UPL', 'SO_UPL', 'SZ_UPL', 'TI_UPL', 'TG_UPL', 'UR_UPL', 'VS_UPL', 'UR_OG']


# courts with "numpy.linalg.LinAlgError: singular matrix" error when creating reports: ["AI_XX", "CH_WBK"]

# NOT CREATED: StopIteration; line 330 dataset_creator.py, load sections
court_error0 = ["CH_VB", "OW_OG", "OW_VG", "OW_VB", "TG_OG", "TI_CRP", "TI_GIAR", "TI_PP", "UR_REB", "ZG_UPL", "BE_UPL",
                "FR_UPL", "VD_UPL", "ZH_UPL", "LU_UPL"]

# NOT CREATED: unexpected EOF while parsing, judgments empty
court_error1 = ["AG_OSB", "VD_SR", 'AG_AK', 'AG_LWRK', 'AG_PRG', 'AG_SKBG', 'AG_SRG']

# NOT CREATED: ValueError: need at least one array to concatenate, line 50: judgment_creator_dataset.py -> df.label
# is empty
court_error2 = ["VS_AG", "ZH_SOBE"]

# NOT CREATED: judgments_df is empty
court_empty = ['AG_XX', 'AI_BZG', 'AR_RR', 'AR_SRK', 'AR_VB', 'AR_XX', 'BE_XX', 'BL_XX', 'BS_XX', 'CH_XX', 'FR_TAMA',
               'FR_XX', 'GE_XX', 'GL_XX', 'GR_XX', 'JU_XX', 'LU_RSH', 'LU_XX', 'NE_XX', 'NW_XX', 'OW_XX', 'SG_XX',
               'SH_XX', 'SO_XX', 'SZ_XX', 'TG_XX', 'TI_GPC', 'TI_XX', 'UR_XX', 'VD_XX', 'VS_XX', 'ZG_XX', 'ZH_XX',
               'CH_UPL', 'AG_UPL', 'AG_RR', 'AG_JL', 'AI_UPL', 'AR_UPL', 'BL_UPL', 'BS_UPL', 'GE_UPL', 'GL_UPL',
               'GR_UPL', 'JU_UPL', 'NE_UPL', 'NW_UPL', 'OW_UPL', 'SG_UPL', 'SH_UPL', 'SO_UPL', 'SZ_UPL', 'TI_UPL',
               'TG_UPL', 'UR_UPL', 'VS_UPL', 'UR_OG', 'AI_KG', 'BE_AK', 'BE_OG', 'CH_EDÖB']


def get_error_courts(index=-1):
    """
    :param index: select a specific error court list, default: all error courts
    :return: list of strings with court names
    """
    error_courts = [court_error0, court_error1, court_error2]
    if index == -1:
        return court_error0 + court_error1 + court_error2
    return error_courts[index]


def get_empty_courts():
    """
    :return: list of strings with court names
    """
    return court_empty
