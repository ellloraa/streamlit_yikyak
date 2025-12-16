import numpy as np
from scipy.sparse import csr_matrix

FIRST_PERSON = {"i","me","my","mine","i'm","im"}
SECOND_PERSON = {"you","your","yours","u","ur","you're","youre"}

def build_numeric_from_text(texts):
    feats = []
    for t in texts:
        t = "" if t is None else str(t)
        words = t.lower().split()

        fp = sum(w in FIRST_PERSON for w in words)
        sp = sum(w in SECOND_PERSON for w in words)
        n_words = max(len(words), 1)

        disagree = 0.0
        conflict = 0.0
        vader = 0.0

        feats.append([
            float(len(t)),          # text_length
            0.0,                    # created_hour
            float(fp),              # first_person_count
            float(sp),              # second_person_count
            float(fp) / n_words,    # first_person_ratio
            float(sp) / n_words,    # second_person_ratio
            float(disagree),        # disagree_count
            float(conflict),        # conflict_count
            float(t.count("!")),    # exclamations
            float(t.count("?")),    # questions
            float(vader),           # vader_compound
            0.0,                    # posts_prev_2h_all
            0.0,                    # rel_posts_prev_2h
            0.0,                    # burst_z_group
            0.0,                    # burst_flag_group
        ])
    return np.array(feats, dtype=float)

def numeric_from_text_transform(X):
    texts = ["" if t is None else str(t) for t in np.asarray(X).ravel()]
    return build_numeric_from_text(texts)

def to_csr_matrix(A):
    return csr_matrix(A)
