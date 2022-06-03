import os
from typing import List, NamedTuple
import numpy as np

DATA_DIR = 'data'
DOWNLOAD_DIR = os.path.join(DATA_DIR, 'raw')

HG_DOWNLOAD_SERVER = 'ftp.ebi.ac.uk'
HG_DOWNLOAD_URL = '/pub/databases/gencode/Gencode_human/release_38/GRCh38.primary_assembly.genome.fa.gz'
HG_DOWNLOAD_TO = os.path.join(DOWNLOAD_DIR, 'hg.fa')

TRAIN_FILE = os.path.join(DATA_DIR, 'data.h5')
CONTEXT_LEN = 10000
PREDICTION_LEN = 5000
INPUT_LEN = CONTEXT_LEN + PREDICTION_LEN

class Fold(NamedTuple):
    chromosomes: List[str]
    train_iterations: int

_all_chroms = [f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY']
_test_chroms = ['chr1', 'chr3', 'chr5', 'chr7', 'chr9']
FOLDS = {
    'ALL': Fold(
        chromosomes = _all_chroms,
        train_iterations = 10
    ),
    'TRAIN': Fold(
        chromosomes = [chrom for chrom in _all_chroms if chrom not in _test_chroms],
        train_iterations = 10
    ),
    'TEST': Fold(
        chromosomes=_test_chroms,
        train_iterations=None
    ),
}

# one-hot configurations
OH_X = np.asarray([[0, 0, 0, 0], # N
                   [1, 0, 0, 0], # A
                   [0, 1, 0, 0], # C
                   [0, 0, 1, 0], # G
                   [0, 0, 0, 1]],# T
                   dtype=bool)

OH_Y = {
    'neither':  np.asarray([1, 0, 0], dtype=bool),
    'acceptor': np.asarray([0, 1, 0], dtype=bool),
    'donor':    np.asarray([0, 0, 1], dtype=bool),
}

class Gencode(NamedTuple):
    download_server: str
    download_url: str
    download_target: str
    annotation_table_path: str # path to the splice table. Used in the pip module for gene annotation but not for training
    train_table_path: str # similar to splice table, but combines overlapping genes into one row. Used for training

GENCODE = [
    Gencode(
        download_server = 'ftp.ebi.ac.uk',
        download_url = '/pub/databases/gencode/Gencode_human/release_37/gencode.v37.annotation.gff3.gz',
        download_target = os.path.join(DOWNLOAD_DIR, 'gencode_GRCh38'),
        annotation_table_path = os.path.join(DATA_DIR, 'splicing_GRCh38.annotation.csv'),
        train_table_path = os.path.join(DATA_DIR, 'splicing_GRCh38.train.csv'),
    ),
    Gencode(
        download_server = 'ftp.ebi.ac.uk',
        download_url = '/pub/databases/gencode/Gencode_human/release_37/GRCh37_mapping/gencode.v37lift37.annotation.gff3.gz',
        download_target = os.path.join(DOWNLOAD_DIR, 'gencode_GRCh37'),
        annotation_table_path = os.path.join(DATA_DIR, 'splicing_GRCh37.annotation.csv'),
        train_table_path = os.path.join(DATA_DIR, 'splicing_GRCh37.train.csv'),
    ),
]

TRAIN_SPLICE_TABLE = GENCODE[0].train_table_path

VALIDATE_SPLIT_SIZE = .1
BATCH_SIZE_PER_GPU = 32
LEARNING_RATE_START = 0.001
LEARNING_RATE_DECAY_FACTOR = .5
LEARNING_RATE_DECAY_AT = [.6,.7,.8,.9]