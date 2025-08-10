# egf/alphafold.py

import argparse
import logging
import math
import numpy as np
import os
import ast
import wandb
import sys
import datetime
import pickle
import random
import time
import torch
import glob
import json
from egf.utils import random_seed
from egf.plot import plot_rmsd_path
from egf.download import download_structure
from openfold.utils.script_utils import (
    load_models_from_command_line,
    parse_fasta,
    run_model,
    prep_output,
    relax_protein, 
)
from omegaconf import OmegaConf
from openfold.config import model_config
from openfold.data import templates, feature_pipeline, data_pipeline
from openfold.np import residue_constants, protein
from openfold.utils.tensor_utils import tensor_tree_map

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(level=logging.INFO)
torch.set_grad_enabled(False)

class AlphaFold:
    def __init__(self, args, guide_config, full_config=None):
        self.guide_config = guide_config
        if (
            hasattr(self.guide_config, "tag_cluster_mapping_path")
            and self.guide_config.tag_cluster_mapping_path is not None
        ):
            with open(self.guide_config.tag_cluster_mapping_path, "r") as f:
                self.tag_cluster_mapping = json.load(f)
        else:
            self.tag_cluster_mapping = None
        self.full_config = full_config
        self.args = args
        self.config = model_config(
            self.args.config_preset,
            long_sequence_inference=self.args.long_sequence_inference,
        )
        self.model_generator = list(
            load_models_from_command_line(
                self.args,
                self.config,
                guide_config=self.guide_config
            )
        )

    def preprocess(self, tag, tags, seqs, data_dirs):
        template_featurizer = templates.HhsearchHitFeaturizer(
            mmcif_dir=data_dirs["template_dir"],
            max_template_date=self.args.max_template_date,
            max_hits=self.config.data.predict.max_templates,
            kalign_binary_path=self.args.kalign_binary_path,
            release_dates_path=self.args.release_dates_path,
            obsolete_pdbs_path=self.args.obsolete_pdbs_path,
        )

        monomer_data_processor = data_pipeline.DataPipeline(
            template_featurizer=template_featurizer,
        )

        is_multimer = isinstance(seqs, list) and len(seqs) > 1
        tmp_fasta_path = os.path.join(data_dirs["output_dir"], f"tmp_{os.getpid()}.fasta")
        if is_multimer:
            with open(tmp_fasta_path, "w") as fp:
                fp.write("\n".join([f">{t}\n{s}" for t, s in zip(tags, seqs)]))
        else:
            with open(tmp_fasta_path, "w") as fp:
                fp.write(f">{tags[0]}\n{seqs[0]}")
        
        # MonomerかMultimerかで処理を分岐。あとでmonomer-maskアリの部分も修正する。
        if is_multimer:
            print("INFO: Running multimer data pipeline")
            data_processor = data_pipeline.DataPipelineMultimer(
                monomer_data_pipeline=monomer_data_processor,
            )
            feature_dict = data_processor.process_fasta(
                fasta_path=tmp_fasta_path, 
                alignment_dir=data_dirs["alignment_dir"]
            )
        else:
            print("INFO: Running monomer data pipeline")
            local_alignment_dir = os.path.join(data_dirs["alignment_dir"], tag)
            feature_dict = monomer_data_processor.process_fasta(
                fasta_path=tmp_fasta_path,
                alignment_dir=local_alignment_dir,
            )
        
        feature_processor = feature_pipeline.FeaturePipeline(self.config.data)
        processed_feature_dict = feature_processor.process_features(
            feature_dict, mode='predict', is_multimer=is_multimer
        )

        processed_feature_dict = {
            k: torch.as_tensor(v, device=self.args.model_device)
            for k, v in processed_feature_dict.items()
        }
        
        os.remove(tmp_fasta_path)
        return feature_dict, processed_feature_dict

    def postprocess(self, feature_dict, processed_feature_dict, out, unrelaxed_output_path, output_directory, tag):
        # --- 1. 必要なライブラリと関数をインポート ---
        from openfold.utils.loss import compute_tm
        import matplotlib.pyplot as plt
    
        # --- 2. スコアの計算と表示 (PyTorchテンソルのまま実行) ---
        plddt = out["plddt"].cpu().numpy()
        mean_plddt = np.mean(plddt)
        
        print("---------------------------------")
        print(f"CONFIDENCE SCORES for {tag}:")
        print(f"  pLDDT: {mean_plddt:.4f}")
    
        is_multimer = "multimer" in self.args.config_preset
        if is_multimer and "predicted_aligned_error" in out:
            pae_logits = out["predicted_aligned_error"]
            if pae_logits.shape[0] == 64:
                pae_logits = pae_logits.permute(1, 2, 0)
            ptm_output = compute_tm(pae_logits, max_bin=31, no_bins=64)
            iptm = ptm_output["iptm"]
            ptm = ptm_output["ptm"]
            
            print(f"  pTM: {ptm.item():.4f}")
            print(f"  ipTM: {iptm.item():.4f}")
    
            pae = torch.nn.functional.softmax(pae_logits, dim=-1) * torch.arange(0, 64, device=pae_logits.device)
            pae = torch.sum(pae, dim=-1).cpu().numpy()
            pae_output_path = os.path.join(output_directory, f"{tag}_pae.npy")
            np.save(pae_output_path, pae)
            print(f"  PAE matrix saved to: {pae_output_path}")
            
            plt.figure(figsize=(10, 8))
            plt.imshow(pae, cmap='viridis_r')
            plt.colorbar(label="Predicted Aligned Error (Å)")
            plt.title(f"PAE for {tag}")
            plt.xlabel("Scored residue")
            plt.ylabel("Aligned residue")
            pae_png_path = os.path.join(output_directory, f"{tag}_pae.png")
            plt.savefig(pae_png_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  PAE heatmap saved to: {pae_png_path}")
    
        print("---------------------------------")
        
        # --- 3. NumPy配列への変換とPDB/CIFオブジェクトの作成 ---
        out_np = tensor_tree_map(lambda x: np.array(x.cpu().to(torch.float32)), out)
        unrelaxed_protein = prep_output(
            out_np,
            processed_feature_dict,
            feature_dict,
            feature_pipeline.FeaturePipeline(self.config.data),
            self.args.config_preset,
            self.args.multimer_ri_gap,
            self.args.subtract_plddt,
            is_multimer=is_multimer,
        )
        
#     def postprocess(self, feature_dict, processed_feature_dict, out, unrelaxed_output_path, output_directory, tag):
#         from openfold.utils.loss import compute_tm
#         import matplotlib.pyplot as plt
#         # processed_feature_dict = tensor_tree_map(
#         #     lambda x: np.array(x[..., -1].cpu()), processed_feature_dict
#         # )
#         out = tensor_tree_map(lambda x: np.array(x.cpu().to(torch.float32)), out)
#         is_multimer = "multimer" in self.args.config_preset

#         unrelaxed_protein = prep_output(
#             out,
#             processed_feature_dict,
#             feature_dict,
#             feature_pipeline.FeaturePipeline(self.config.data), 
#             self.args.config_preset,
#             self.args.multimer_ri_gap,
#             self.args.subtract_plddt,
#             is_multimer=is_multimer,
# #            is_multimer = "multimer" in self.args.config_preset
# #            is_multimer=isinstance(feature_dict["msa"], list) 
#         )

#         plddt = out["plddt"]
#         mean_plddt = np.mean(plddt)
        
#         print("---------------------------------")
#         print(f"CONFIDENCE SCORES for {tag}:")
#         print(f"  pLDDT: {mean_plddt:.4f}")
        
#         if is_multimer and "predicted_aligned_error" in out:
#             # PAEのlogitsを取得
#             pae_logits = out["predicted_aligned_error"]
            
#             # ipTMとpTMを計算
#             ptm_output = compute_tm(
#                 pae_logits,
#                 max_bin=31,
#                 no_bins=64
#             )
#             iptm = ptm_output["iptm"]
#             ptm = ptm_output["ptm"]
            
#             # 結果をログに出力
#             print(f"  pTM: {ptm.item():.4f}")
#             print(f"  ipTM: {iptm.item():.4f}")
        
#             # PAEを計算してファイルに保存
#             pae = torch.nn.functional.softmax(pae_logits, dim=-1) * torch.arange(0, 64, device=pae_logits.device)
#             pae = torch.sum(pae, dim=-1).cpu().numpy()
#             pae_output_path = os.path.join(output_directory, f"{tag}_pae.npy")
#             np.save(pae_output_path, pae)
#             print(f"  PAE matrix saved to: {pae_output_path}")
            
#             # PAEをヒートマップとして画像保存
#             plt.figure(figsize=(10, 8))
#             plt.imshow(pae, cmap='viridis_r')
#             plt.colorbar(label="Predicted Aligned Error (Å)")
#             plt.title(f"PAE for {tag}")
#             plt.xlabel("Scored residue")
#             plt.ylabel("Aligned residue")
#             pae_png_path = os.path.join(output_directory, f"{tag}_pae.png")
#             plt.savefig(pae_png_path, dpi=300, bbox_inches='tight')
#             plt.close()
#             print(f"  PAE heatmap saved to: {pae_png_path}")
        
#         print("---------------------------------")

        if unrelaxed_output_path.endswith('.cif'):
            unrelaxed_output_path = unrelaxed_output_path.replace(".cif", ".pdb")
        with open(unrelaxed_output_path, "w") as fp:
            fp.write(protein.to_pdb(unrelaxed_protein))

        logger.info(f"Final PDB output written to {unrelaxed_output_path}")

#        .cifだと書き出しが出来なくなる場合が存在するので基本的には.pdb
#        if self.args.cif_output:
#            cif_output_path = unrelaxed_output_path.replace(".pdb", ".cif")
#            with open(cif_output_path, "w") as fp:
#                fp.write(protein.to_modelcif(unrelaxed_protein))
#            unrelaxed_output_path = cif_output_path
#        else:
#            with open(unrelaxed_output_path, "w") as fp:
#                fp.write(protein.to_pdb(unrelaxed_protein))

        if not self.args.skip_relaxation:
            logger.info(f"Running relaxation on {unrelaxed_output_path}...")
            relax_protein(
                self.config,
                self.args.model_device,
                unrelaxed_protein,
                output_directory,
                tag,
                cif_output=False,
            )

    def run(self, input_dict, data_dirs):
        output_dir_base = data_dirs["output_dir"]
        os.makedirs(output_dir_base, exist_ok=True)
        
        output_paths = {}
        for model, _ in self.model_generator:

            config_output_path = os.path.join(output_dir_base, "config.yaml")
            with open(config_output_path, "w") as f:
                OmegaConf.save(self.full_config, f) 

            for fasta_filename, fasta_content in input_dict.items():
                tags, seqs = parse_fasta(fasta_content)
                tag = "-".join(tags)

                if self.tag_cluster_mapping is not None and tag not in self.tag_cluster_mapping:
                    print(f"Skipping {tag} as it's not in the tag_cluster_mapping.json")
                    continue
                
                print(f"INFO: Processing target {tag}")
                
                random_seed(self.args.data_random_seed)
                unrelaxed_output_path = os.path.join(
                    output_dir_base,
                    f"{tag}.cif" if self.args.cif_output else f"{tag}.pdb",
                )
                if (
                    os.path.exists(unrelaxed_output_path)
                    and self.guide_config.skip_existing
                ):
                    continue

                feature_dict, processed_feature_dict = self.preprocess(
                    tag, tags, seqs, data_dirs
                )

                out, intermediate_outputs, intermediate_metrics = run_model(
                    model, processed_feature_dict, tag, output_dir_base
                )

                if self.guide_config.info_dir is not None:
                    os.makedirs(self.guide_config.info_dir, exist_ok=True)
                    metrics_save_path = os.path.join(
                        self.guide_config.info_dir, f"{tag}_metrics.json"
                    )
                    with open(metrics_save_path, "w") as f:
                        json.dump(intermediate_metrics, f)

                self.postprocess(
                    feature_dict,
                    processed_feature_dict,
                    out,
                    unrelaxed_output_path,
                    output_dir_base,
                    tag,
                )
                logger.info(f"Output written to {unrelaxed_output_path}...")
        
        return output_paths

def get_base_model(config):
    return AlphaFold(config.base, config.guide_config, full_config=config)
