# process_pdbs.py
import numpy as np
import torch
from Bio.PDB import PDBParser
import sys
import os

# (関数の定義は省略... 以前の回答と同じコード)

def get_atom_coords(residue, atom_name):
    """アミノ酸残基オブジェクトから特定原子の座標を取得"""
    if atom_name in residue:
        return residue[atom_name].get_coord()
    elif 'CA' in residue: # フォールバックとしてCα原子
        return residue['CA'].get_coord()
    return None

def create_truth_distogram(pdb_path, output_path):
    """PDBファイルから正解ディストグラムを作成し、.npyとして保存する"""
    print(f"Loading PDB: {pdb_path}")
    bin_edges = np.linspace(2, 22, num=63)
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("protein", pdb_path)
    except Exception as e:
        print(f"  - Error parsing PDB file: {e}")
        return
        
    model = structure[0]
    chain = next(model.get_chains())
    residues = [res for res in chain if "CA" in res]
    n_res = len(residues)

    dist_matrix = np.zeros((n_res, n_res))
    for i in range(n_res):
        for j in range(n_res):
            atom1_name = 'CB' if residues[i].get_resname() != 'GLY' else 'CA'
            atom2_name = 'CB' if residues[j].get_resname() != 'GLY' else 'CA'
            
            coord1 = get_atom_coords(residues[i], atom1_name)
            coord2 = get_atom_coords(residues[j], atom2_name)

            if coord1 is not None and coord2 is not None:
                dist_matrix[i, j] = np.linalg.norm(coord1 - coord2)
            else:
                dist_matrix[i, j] = 999

    binned_distances = np.digitize(dist_matrix, bins=bin_edges)
    truth_distogram = np.eye(64)[binned_distances]

    print(f"  - Saving ground truth distogram to: {output_path}")
    np.save(output_path, truth_distogram)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("使い方: python process_pdbs.py <input_pdb_path> <output_npy_path>")
        sys.exit(1)
    
    create_truth_distogram(sys.argv[1], sys.argv[2])
