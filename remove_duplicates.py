import pandas as pd
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm
import argparse
from pprint import pprint

def range_limited_float_type(arg,MIN_VAL=0.0,MAX_VAL=1.0):
    """ Type function for argparse - a float within some predefined bounds """
    try:
        f = float(arg)
    except ValueError:    
        raise argparse.ArgumentTypeError("Must be a floating point number")
    if f < MIN_VAL or f > MAX_VAL:
        raise argparse.ArgumentTypeError("Argument must be < " + str(MAX_VAL) + "and > " + str(MIN_VAL))
    return f


parser = argparse.ArgumentParser(description='Script to remove duplicates from csv')
parser.add_argument('--num_perm',type=int,default=256,help='Number of permutations to use in LSH during minhash creation')
parser.add_argument('--min_len',type=int,default=5,help="All sentences below this will be automatically removed")
parser.add_argument('--save_dup_info',type=bool,default=True,help="Saves a dictionary with key being the index of a row of a df and value being the list of indexes that are duplicates to that row")
parser.add_argument('--threshold',type=range_limited_float_type,default=0.6,help='Jaccard Similarity Threshold above which text will be assumed duplicate')
parser.add_argument('--fname',type=str,default='train.csv',help='CSV file name to remove duplicates from')
parser.add_argument('--text_col_name',type=str,default='bullet',help='Column name for which duplicates need to be found')
parser.add_argument('--out_fname',type=str,default='clean.csv',help='Output filename')
parser.add_argument('--dup_info_fname',type=str,default='duplicates.txt',help='Duplicates info filename')


def length_check(text,min_len=5):
    l = len(text.split())
    if l > min_len:
        return False
    else:
        return True

def dedupe(df,lsh,min_len=5):
    to_remove = set()
    to_remove_dict = dict()
    for idx, row in tqdm(df.iterrows()):
        index = row["index"]
        if length_check(row["bullet"],min_len=min_len):
            to_remove.add(index)
        if index in to_remove:
            continue
        sim_rows = sorted(lsh.query(row["mhash"]))
        if len(sim_rows) > 1:
            to_remove_dict[index] = sim_rows[1:]
            to_remove.update(sim_rows[1:])
    to_remove = list(to_remove)
    print(f'Duplicates Found {len(to_remove)}')
    df = df.drop(to_remove, axis='index')
    df = df.drop(["mhash"], axis=1)
    return df, to_remove, to_remove_dict

def preprocess(text):
    return text.encode('utf-8').lower().split()

def create_min_hashes(texts, num_perm=256):
    texts = list(map(preprocess, texts))
    mhashs = MinHash.bulk(texts, num_perm=num_perm)
    return mhashs

def main(args):
    df = pd.read_csv(args.fname)
    df= df.reset_index()
    bullets = list(df[args.text_col_name])
    
    lsh = MinHashLSH(threshold=args.threshold, num_perm=args.num_perm)
    
    print(f"Creating MinHashes")
    mhash = create_min_hashes(bullets, num_perm=args.num_perm)
    df["mhash"] = mhash
    
    print(f"Inserting in LSH")
    with lsh.insertion_session() as session:
        for idx, row in tqdm(df.iterrows()):
            session.insert(row["index"], row["mhash"])
    
    print(f"Cleaning Dataframe")
    df, to_remove, to_remove_dict = dedupe(df, lsh,min_len=args.min_len)
    if args.save_dup_info:
        with open('duplicates.txt','w') as out:
            pprint(to_remove_dict,stream=out)
    df = df.drop(["index"],axis=1)
    df.to_csv(args.out_fname,index=False,header=False)
    
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)