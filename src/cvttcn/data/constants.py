"""Static facts about the EEGMMIDB (EEG Motor Movement/Imagery) dataset.

These describe the recording itself, not tunable choices (those live in
``cvttcn.config``). A unit test asserts the config defaults stay consistent with
these values so the two never silently drift apart.

Reference: PhysioNet EEG Motor Movement/Imagery Database
(https://physionet.org/content/eegmmidb/1.0.0/), recorded with BCI2000.
"""

# Recording properties.
N_SUBJECTS = 109          # subjects are numbered 1..109
N_CHANNELS = 64           # EEG channels
SFREQ = 160.0             # nominal sampling rate (Hz)

# EEGMMIDB run taxonomy. Every subject has 14 runs:
BASELINE_RUNS = (1, 2)                  # 1: eyes open, 2: eyes closed
REAL_FIST_RUNS = (3, 7, 11)             # executed open/close of left or right fist
IMAGINED_FIST_RUNS = (4, 8, 12)         # imagined open/close of left or right fist
REAL_HANDS_FEET_RUNS = (5, 9, 13)       # executed both fists or both feet
IMAGINED_HANDS_FEET_RUNS = (6, 10, 14)  # imagined both fists or both feet

# Subjects whose records have inconsistent annotation timing / sampling rate and
# are conventionally excluded from analyses.
DEFAULT_EXCLUDED_SUBJECTS = (88, 89, 92, 100)

# This project classifies imagined left vs right fist movement, which lives in
# IMAGINED_FIST_RUNS. Within those runs the EDF annotations are:
#   T0 -> rest (dropped for the binary task)
#   T1 -> left fist
#   T2 -> right fist
# The annotation -> integer-label mapping is applied during preprocessing
# (next commit), since it belongs to epoch extraction rather than acquisition.
