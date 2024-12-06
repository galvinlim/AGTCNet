from . import config

def subject_run_map(DATASET, SUBJECT_SELECTION, SESSION_SELECTION, SUBJECT, VALID_RUN):
    if DATASET == 'BCICIV2A':
        SUBJECT_LIST = [s for s in range(1, 10)]
        RUN_LIST = [False, True]
    elif DATASET == 'EEGMMIDB':
        SUBJECT_LIST = [s for s in range(1, 110) if s not in config.dataset_EEGMMIDB.subject_exemption]
        RUN_LIST = [4, 6, 8, 10, 12, 14]
        
    if SUBJECT_SELECTION == 'SL':
        TRAIN_SUBJECT = VALID_SUBJECT = SUBJECT
    elif SUBJECT_SELECTION == 'SM':
        TRAIN_SUBJECT = VALID_SUBJECT = SUBJECT_LIST
    elif SUBJECT_SELECTION == 'SN':
        TRAIN_SUBJECT = [s for s in SUBJECT_LIST if s not in SUBJECT]
        VALID_SUBJECT = [s for s in SUBJECT if s in SUBJECT_LIST]

    if SESSION_SELECTION == 'RS' or SUBJECT_SELECTION == 'SN':
        TRAIN_RUN = VALID_RUN = RUN_LIST
    else:
        TRAIN_RUN = [r for r in RUN_LIST if r not in VALID_RUN]
        VALID_RUN = VALID_RUN

    return (TRAIN_SUBJECT, TRAIN_RUN), (VALID_SUBJECT, VALID_RUN)
