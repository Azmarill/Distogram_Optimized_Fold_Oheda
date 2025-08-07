# create_truth_distances.py
import numpy as np
from Bio.PDB import PDBParser
import sys
import os

def get_atom_coords(residue, atom_name):
    if atom_name in residue:
        return residue[atom_name].get_coord()
    elif 'CA' in residue:
        return residue['CA'].get_coord()
    return None

def create_truth_distances(pdb_path, output_path):
    print(f"Loading PDB: {pdb_path}")
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
                dist_matrix[i, j] = 999 # mask value

    print(f"  - Saving raw distance matrix to: {output_path}")
    np.save(output_path, dist_matrix)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("使い方: python create_truth_distances.py <input_pdb_path> <output_npy_path>")
        sys.exit(1)
    
    create_truth_distances(sys.argv[1], sys.argv[2])
