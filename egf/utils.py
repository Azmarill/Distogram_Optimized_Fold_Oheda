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

import glob
import os

def load_fasta_sequences(fasta_dir: str) -> dict:
    """
    指定されたディレクトリ内のFASTAファイルを読み込み、
    {ファイル名: ファイルの中身（文字列）} の辞書を返す
    """
    input_dict = {}
    for fasta_file in glob.glob(os.path.join(fasta_dir, "*.fasta")):
        with open(fasta_file, "r") as f:
            data = f.read()
        
        main_tag = os.path.splitext(os.path.basename(fasta_file))[0]
        input_dict[main_tag] = data
            
    return input_dict
