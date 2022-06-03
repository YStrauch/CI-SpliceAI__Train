from __future__ import division

import numpy as np
from mpi4py import MPI
import h5py
import pandas as pd
import lib.networking as networking
from pyfaidx import Fasta
import env
import splicing_table

from lib.progress import ProgressOutput

# calculate the number of data points in the h5 file
n_chunks, df = splicing_table.get_train_splice_table()

n_points = df['slices_in_gene'].agg(sum)

rank = MPI.COMM_WORLD.rank
print('Worker %d of %d starting' % (rank+1, MPI.COMM_WORLD.size))

out = h5py.File(env.TRAIN_FILE, 'w', driver='mpio', comm=MPI.COMM_WORLD)
out_x = out.create_dataset('X', (n_points, 15000, 4), dtype='bool')
out_y = out.create_dataset('Y', (n_points, 5000, 3), dtype='bool')

# -- DOWNLOAD REFERENCE GENOME --
networking.download_extract_gz_from_ftp(env.HG_DOWNLOAD_SERVER, env.HG_DOWNLOAD_URL, env.HG_DOWNLOAD_TO)

# make sure that only the first process builds the index
comm = MPI.COMM_WORLD
if rank == 0:
    fasta = Fasta(env.HG_DOWNLOAD_TO)
    req = comm.barrier()
else:
    req = comm.barrier()
    fasta = Fasta(env.HG_DOWNLOAD_TO, rebuild=False)


# -- LOGIC TO EXTRACT AND PARSE DATA --

# convert comma-separated junction list to list of ints
def explode_jn(jns):
    return list(map(int, jns.split(',')))

df.jn_start = df.jn_start.map(explode_jn)
df.jn_end = df.jn_end.map(explode_jn)

def create_x(gene):
    '''Extracts sequence for a gene and encodes it one-hot. Reverse-complements if needed.'''
    # -1 for start because fasta uses 0-index; not -1 for end to include the last nucleotide of the gene
    seq = fasta[gene.chr][gene.start-1:gene.end].seq

    # to numeric - reverse-complement if needed
    if gene.strand == '+':
        num = seq.upper().replace('N', '0').replace('A', '1').replace('C', '2').replace('G', '3').replace('T', '4')
    else:
        num = seq.upper().replace('A', '4').replace('C', '3').replace('G', '2').replace('T', '1').replace('N', '0')[::-1]

    num = np.asarray(list(map(int, list(num))))
    
    # one-hot encode
    oh = env.OH_X[num]
    return oh

def create_y(gene):
    '''Creates a one-hot encoded array with annotated ground truth (neither/acceptor/donor). Reverses if needed.'''
    if gene.strand == '+':
        donors, acceptors = gene.jn_start, gene.jn_end
    else:
        donors, acceptors = gene.jn_end, gene.jn_start

    y = np.array([env.OH_Y['neither']] * (gene.end - gene.start + 1)) # +1 to include last nucleotide of gene
    for site in donors:
        y[site - gene.start] = env.OH_Y['donor']
    for site in acceptors:
        y[site - gene.start] = env.OH_Y['acceptor']
    
    if gene.strand == '-':
        return y[::-1]
    return y


def process_gene(gene, h5_x, h5_y):
    x = create_x(gene)
    y = create_y(gene)

    assert len(x) == len(y)

    # add masked context to gene: 5k on either side (fill with all-zero vector, or "N")
    flank_x = np.zeros((env.CONTEXT_LEN//2,4), dtype=bool)

    x = np.concatenate([flank_x, x, flank_x])

    assert np.ceil(len(y)/env.PREDICTION_LEN) == gene.slices_in_gene, f'Expected {gene.slices_in_gene} slices in gene {gene.name}, got {np.ceil(len(y)/env.PREDICTION_LEN)}'

    for slice in range(gene.slices_in_gene):
        start_slice = slice * env.PREDICTION_LEN           
        # extract gene slice
        x_slice = x[start_slice : start_slice + env.INPUT_LEN]
        y_slice = y[start_slice : start_slice + env.PREDICTION_LEN]

        # pad gene slice on the right if it's too short (i.e. last slice in gene or gene too short)
        pad_len = env.INPUT_LEN - len(x_slice)
        if pad_len:
            x_slice = np.concatenate([x_slice, np.zeros((pad_len, 4), dtype=bool)])
            y_slice = np.concatenate([y_slice, np.zeros((pad_len, 3), dtype=bool)])
        
        assert len(x_slice) == env.INPUT_LEN
        assert len(y_slice) == env.PREDICTION_LEN

        h5_x[gene.slices_before+slice] = x_slice
        h5_y[gene.slices_before+slice] = y_slice



progress = ProgressOutput(len(df))
append_index = 0
for i, (id, gene) in enumerate(df.iterrows()):
    assert append_index == gene.slices_before

    if rank == 0:
        progress.update(i)
    
    if i % MPI.COMM_WORLD.size == rank:
        # this is our gene to process
        try:
            process_gene(gene, out_x, out_y)
        except Exception as e:
            out.close()
            raise e
    
    append_index += gene.slices_in_gene

assert append_index == len(out_x)
out.close()


if rank == 0:
    progress.update(len(df))