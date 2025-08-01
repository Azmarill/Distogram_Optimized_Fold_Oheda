import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import os
import random

import glob
import os
from openfold.utils.script_utils import parse_fasta

def random_seed(seed):
    if seed is None:
        seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def list_files_with_extensions(dir, extensions):
    return [f for f in os.listdir(dir) if f.endswith(extensions)]


#def load_fasta_sequences(fasta_dir):
#    from openfold.utils.script_utils import parse_fasta

#    all_sequences = {}
#    if os.path.exists(fasta_dir) and fasta_dir.endswith(".fasta"):
#        with open(fasta_dir, "r") as fp:
#            data = fp.read()
#        tags, seqs = parse_fasta(data)
#        for tag, seq in zip(tags, seqs):
#            all_sequences[tag] = seq
#        return all_sequences

#    for fasta_file in list_files_with_extensions(fasta_dir, (".fasta", ".fa")):
        # Gather input sequences
#        with open(os.path.join(fasta_dir, fasta_file), "r") as fp:
#            data = fp.read()

#        tags, seqs = parse_fasta(data)
#        for tag, seq in zip(tags, seqs):
#            all_sequences[tag] = seq
#    return all_sequences

# egf/utils.py の load_fasta_sequences 関数を以下に置き換え

def load_fasta_sequences(fasta_dir: str) -> dict:
    """
    指定されたディレクトリ内のFASTAファイルを読み込む。
    ファイル内に1つの配列しかない場合は単量体として、
    複数の配列がある場合は多量体として扱う。
    """
    input_dict = {}
    for fasta_file in glob.glob(os.path.join(fasta_dir, "*.fasta")):
        with open(fasta_file, "r") as f:
            data = f.read()

        # OpenFoldの高性能なパーサーを使用
        tags, seqs = parse_fasta(data)

        # ファイル名(拡張子なし)を複合体のメインタグとして使用
        main_tag = os.path.splitext(os.path.basename(fasta_file))[0]
        input_dict[main_tag] = (tags, seqs)

    return input_dict
