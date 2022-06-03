'''Creates the splicing tables with collapsed isoforms. Downloads all necessary resources.'''

import itertools
import multiprocessing
import os
import lib.networking as networking
import lib.gff as gff
import pandas as pd
import env
import numpy as np
import h5py

def enlarge_tx(args):
    '''enlarges all transcripts in a gene to the min/max'''
    _, transcripts = args
    transcripts.start = min(transcripts.start)
    transcripts.end = max(transcripts.end)
    return transcripts

def add_exon(args):
    (_, gene), df = args
    mapping = (df['gene_id'] == gene['gene_id']) & (df.chr == gene.chr)
    
    # Put start and end of exons into one array each
    exon_starts, exon_ends = [list(map(int, sorted(df[mapping][i].dropna().drop_duplicates()))) for i in ['start_exon', 'end_exon']]

    # Remove sites that are acceptor -and- donor (i.e. chr7:44122282)
    both = set(exon_ends).intersection(set(exon_starts))
    if both:
        for pos in both:
            print(f'Site is acceptor and donor: {gene.chr}:{pos}')

        exon_ends = [site for site in exon_ends if site not in both]
        exon_starts = [site for site in exon_starts if site not in both]
    
    gene['jn_start'] = ','.join(map(str, exon_ends))
    gene['jn_end'] = ','.join(map(str,exon_starts))
    
    return gene

def create_splice_table(gencode: env.Gencode):
    # -- DOWNLOAD, FILTER, SPLIT GENCODE ANNOTATIONS --
    exons_file = os.path.join(gencode.download_target, 'exons.csv')
    if not os.path.exists(exons_file):
        print('Download files')

        all_file = os.path.join(gencode.download_target, 'all.gff3')
        networking.download_extract_gz_from_ftp(gencode.download_server, gencode.download_url, all_file)

        df = gff.load_gff3(all_file, filter=['chr', 'start', 'end', 'strand', 'feature'], explode_info=False)
        
        # filter to transcript_type protein coding only
        df = df[df['info'].str.contains('transcript_type=protein_coding')]

        # filter to chromosomes 1-22 and X,Y only
        allowed = pd.Series(['chr%s' % d for d in list(range(1, 23)) + ['X', 'Y']])
        df = df[df.chr.isin(allowed)].reset_index(drop=True)

        # explode info column into separate columns
        df = gff.explode_col(df, 'info')

        # remove version from gene id
        df['gene_id'] = df['gene_id'].str.extract(r'([^\n]*)\.\d')

        # split into transcripts and exons
        transcripts = df[df.feature == 'transcript'].drop(columns=['feature']).reset_index(drop=True)
        exons = df[df.feature == 'exon'].drop(columns=['feature']).reset_index(drop=True)

        transcripts.to_csv(os.path.join(gencode.download_target, 'transcripts.csv'), index=False)
        exons.to_csv(os.path.join(gencode.download_target, 'exons.csv'), index=False)

        os.remove(all_file)
    else:
        transcripts = pd.read_csv(os.path.join(gencode.download_target, 'transcripts.csv'))
        exons = pd.read_csv(os.path.join(gencode.download_target, 'exons.csv'))


    # -- CREATE SPLICE TABLE --
    print(f'Create splicing annotation table {gencode.annotation_table_path}')

    # Filter to only human validated transcripts
    levels = pd.to_numeric(transcripts['level'], errors='coerce')
    transcripts = transcripts[(levels <= 2)]

    # Merge transcripts and exons
    df = transcripts.merge(exons, on=['chr', 'gene_id', 'transcript_id'], suffixes=['_transcript', '_exon'], how='inner')
    del transcripts
    del exons

    df = df.rename(columns={
        'strand_transcript': 'strand',
        'start_transcript': 'start',
        'end_transcript': 'end',
    })


    # Multi-process pool
    with multiprocessing.Pool() as pool:
        # -- COLLAPSE ISOFORMS --

        # Remove Exon Boundaries that overlap with Transcript start/end
        df.loc[df.start == df.start_exon, 'start_exon'] = None
        df.loc[df.end == df.end_exon, 'end_exon'] = None

        df.loc[df.end == df.start_exon, 'start_exon'] = None
        df.loc[df.start == df.end_exon, 'end_exon'] = None

        # Transcripts start at different indices per gene
        # Extract where the first isoform starts and the last one ends
        genes = pool.map(enlarge_tx, df.groupby(['chr', 'gene_id']))
        df = pd.concat(genes).reset_index(drop=True)

        # Create splicing table
        out = df.filter(['gene_id', 'chr', 'strand', 'start', 'end']).drop_duplicates()
        genes = pool.map(add_exon, zip(out.iterrows(), itertools.repeat(df)))
        out = pd.DataFrame(genes)#.set_index('gene_id')
        del genes

        # Remove Table Lines without junctions
        out = out[out.jn_start != '']
        out = out[out.jn_end != '']

        # Filter out duplicate gene IDs between X and Y chromosome
        duplicates = out[out.duplicated('gene_id', keep='first')]
        # print duplicates
        assert len(duplicates) == 0 or duplicates.chr.unique() == ['chrY']
        out = out.drop_duplicates('gene_id', keep='first')

        # sorting: sort -numerically- by chrom
        out['chr_num'] = out['chr'].str.slice(3)
        out.loc[out.chr_num == 'X', 'chr_num'] = 23
        out.loc[out.chr_num == 'Y', 'chr_num'] = 24
        out['chr_num'] = out['chr_num'].astype(int)

        out = out.sort_values(['chr_num', 'start', 'end', 'jn_start', 'jn_end']).drop(columns=['chr_num']).set_index('gene_id')

        # write it to the annotation splice table. This will be used by the pip module to annotate most affected genes
        out.to_csv(gencode.annotation_table_path)

    # only proceed to create the training table if we're actually using it
    if gencode.train_table_path != env.TRAIN_SPLICE_TABLE:
        return

    # now, create the train annotation table
    print(f'Create splicing training table {gencode.train_table_path}')

    # Add paralog annotations
    paralogs = pd.read_csv(os.path.join('data', 'paralogs_GRCh38.csv'))
    paralogs['Chromosome/scaffold name'] = 'chr' + paralogs['Chromosome/scaffold name'].astype('str')
    # Convert paralog string to boolean and drop duplicates
    paralogs['Human paralogue homology type'] = pd.notna(paralogs['Human paralogue homology type'])
    paralogs = paralogs.drop_duplicates().set_index(['Chromosome/scaffold name', 'Gene stable ID'])['Human paralogue homology type']
    out = out.join(paralogs.rename('is_paralog'), on=['chr', 'gene_id'])
    print('Warning: Mismatch between paralog annotation and gencode in %d genes' % (sum(out.is_paralog.isna())))
    out['is_paralog'] = out['is_paralog'].fillna(False).astype(np.bool)

    out.to_csv(gencode.train_table_path)

def get_train_splice_table(fold: env.Fold=env.FOLDS['ALL'], drop_paralogs=False):
    splice_table = pd.read_csv(env.TRAIN_SPLICE_TABLE).set_index('gene_id')

    # calculate the number of data slices in the h5 file
    # each slice is 5000 nucleotides long (env.PREDICTION_LEN)
    # increase gene length by one to include last nucleotide
    splice_table['slices_in_gene'] = ((splice_table.end - splice_table.start + 1)/env.PREDICTION_LEN).apply(np.ceil).astype(int)
    # the cumulative number of slices before each gene starts in the h5 file
    splice_table['slices_before'] = splice_table.slices_in_gene.cumsum() - splice_table.slices_in_gene

    # filter for chromosomes in current Fold
    splice_table = splice_table[splice_table.chr.isin(fold.chromosomes)]
    if drop_paralogs:
        splice_table = splice_table[~splice_table.is_paralog]
    splice_table = splice_table.copy() # copy to suppress SettingWithCopyWarning when adding chunks 

    # create gene chunks - 100 genes per chunk (except for last chunk which is smaller for now)
    splice_table['chunk'] = np.arange(len(splice_table)) // 100
    # join last chunk with the previous one to prevent very small chunks
    n_chunks = len(splice_table)//100
    splice_table.loc[splice_table.chunk == n_chunks, 'chunk'] = n_chunks - 1

    return n_chunks, splice_table

def slices_in_chunk(splice_table: pd.DataFrame, h5f:h5py.File, n_gpus: int, chunk):
    genes = splice_table[splice_table.chunk == chunk]
    n_slices = genes.slices_in_gene.sum()

    X = np.zeros((n_slices,15000,4), dtype=np.float32)
    Y = np.zeros((n_slices,5000,3), dtype=np.float32)

    i = 0 # chunk offset in output array
    for _, gene in genes.iterrows():
        next_i = i+gene.slices_in_gene
        X[i:next_i] = h5f['X'][gene.slices_before:gene.slices_before+gene.slices_in_gene]
        Y[i:next_i] = h5f['Y'][gene.slices_before:gene.slices_before+gene.slices_in_gene]
        i = next_i

    # assure that the number of data points is dividable by the number of GPUs
    drop = len(X) % n_gpus
    if drop:
        # drop random data points to make it evenly dividable
        drop = np.random.choice(range(len(X)), drop, replace=False)
        return np.delete(X, drop, axis=0), [np.delete(Y, drop, axis=0)]

    return X, [Y]


if __name__ == '__main__':
    for gencode in env.GENCODE:
        create_splice_table(gencode)