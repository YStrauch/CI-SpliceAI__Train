import pandas as pd

def explode_col(df, field):
    dicts = df[field].map(lambda info: dict(map(lambda e: e.split('='), info.split(';'))))
    data = pd.DataFrame(list(dicts))
    
    df = pd.concat([df,data], axis=1)
    return df.drop(field, axis=1)


def load_gff3(file, columns=['chr', 'annotation', 'feature', 'start', 'end', 'score', 'strand', 'phase' , 'info'], filter=['chr', 'start', 'end', 'strand', 'gene_id', 'transcript_id', 'feature'], explode_info = True):
    
    if not explode_info:
        filter.append('info')

    df = pd.read_csv(file, names=columns, delimiter='\t', comment='#')
    if explode_info:
        df = explode_col(df, 'info')

    return df.filter(filter, axis=1)