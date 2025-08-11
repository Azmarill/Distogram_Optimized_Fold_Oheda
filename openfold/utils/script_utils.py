import json
import logging
import os
import re
import time

import numpy
import torch

from openfold.model.model import AlphaFold
from openfold.np import residue_constants, protein

#from openfold.np.relax import relax
from openfold.utils.import_weights import import_jax_weights_, import_openfold_weights_

from pytorch_lightning.utilities.deepspeed import (
    convert_zero_checkpoint_to_fp32_state_dict,
)

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(level=logging.INFO)


def count_models_to_evaluate(openfold_checkpoint_path, jax_param_path):
    model_count = 0
    if openfold_checkpoint_path:
        model_count += len(openfold_checkpoint_path.split(","))
    if jax_param_path:
        model_count += len(jax_param_path.split(","))
    return model_count


def get_model_basename(model_path):
    return os.path.splitext(os.path.basename(os.path.normpath(model_path)))[0]


def make_output_directory(output_dir, model_name, multiple_model_mode):
    if multiple_model_mode:
        prediction_dir = os.path.join(output_dir, "predictions", model_name)
    else:
        prediction_dir = os.path.join(output_dir, "predictions")
    os.makedirs(prediction_dir, exist_ok=True)
    return prediction_dir

def load_models_from_command_line(
    args,                
    config,              
    guide_config=None
):
    model_device = args.model_device
    jax_param_path = args.jax_param_path
    openfold_checkpoint_path = args.openfold_checkpoint_path
    output_dir = args.output_dir if hasattr(args, 'output_dir') else ""

    if(model_device == "cpu"):
        fp16 = False
        bf16 = False
    else:
        fp16 = True
        bf16 = False
        if(torch.cuda.is_bf16_supported()):
            bf16 = True
            fp16 = False
            
    if(jax_param_path is None and openfold_checkpoint_path is None):
        raise ValueError(
            "At least one of jax_param_path or openfold_checkpoint_path must be specified."
        )

    model = AlphaFold(config, guide_config=guide_config)

    if(openfold_checkpoint_path is not None):
        d = torch.load(openfold_checkpoint_path)
        model.load_state_dict(d)

    model = model.eval()

    if(bf16):
        model = model.bfloat16()
    elif(fp16):
        model = model.half()

    model = model.to(model_device)
    
    if(jax_param_path is not None):
        model_preset = args.config_preset 
        model_version = model_preset
        
        if(not config.globals.is_multimer):
            model_version = model_version.replace('ptm', 'ptm')
        
        import_jax_weights_(
            model, jax_param_path, version=model_version
        )
        logger.info(f"Successfully loaded JAX parameters at {jax_param_path}...")
    
    yield model, output_dir
    
def parse_fasta(data):
    data = re.sub(">$", "", data, flags=re.M)
    lines = [
        l.replace("\n", "")
        for prot in data.split(">")
        for l in prot.strip().split("\n", 1)
    ][1:]
    tags, seqs = lines[::2], lines[1::2]

    tags = [re.split("\W| \|", t)[0] for t in tags]

    return tags, seqs


def update_timings(timing_dict, output_file=os.path.join(os.getcwd(), "timings.json")):
    """
    Write dictionary of one or more run step times to a file
    """
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            try:
                timings = json.load(f)
            except json.JSONDecodeError:
                logger.info(f"Overwriting non-standard JSON in {output_file}.")
                timings = {}
    else:
        timings = {}
    timings.update(timing_dict)
    with open(output_file, "w") as f:
        json.dump(timings, f)
    return output_file


def run_model(model, batch, tag, output_dir):
    with torch.no_grad():
        template_enabled = model.config.template.enabled
        model.config.template.enabled = template_enabled and any(
            ["template_" in k for k in batch]
        )

        logger.info(f"Running inference for {tag}...")
        t = time.perf_counter()
        out = model(batch, tag=tag)
        inference_time = time.perf_counter() - t
        logger.info(f"Inference time: {inference_time}")
        update_timings(
            {tag: {"inference": inference_time}},
            os.path.join(output_dir, "timings.json"),
        )

        model.config.template.enabled = template_enabled

    return out


def prep_output(
    out,
    processed_feature_dict,
    feature_dict,
    feature_processor,
    config_preset,
    multimer_ri_gap,
    subtract_plddt,
    is_multimer=False,
):
    """
    Post-processes the output of the model to be saved to a PDB file.
    """
    plddt = out["plddt"]
    if subtract_plddt:
        plddt = plddt - out["plddt_baseline"]
    
    plddt_b_factors = np.repeat(
        plddt[..., None], residue_constants.atom_type_num, axis=-1
    )

    # --- ▼▼▼ ここが最終的な修正部分 ▼▼▼ ---
    # aatype と residue_index は、モデルに入力する前の生の feature_dict から取得する
    final_aatype = feature_dict["aatype"]
    final_residue_index = feature_dict["residue_index"] + 1

    if is_multimer:
        # chain_index も生の feature_dict から取得する
        final_chain_index = feature_dict["chain_index"]

        return protein.Protein(
            atom_positions=out["final_atom_positions"],
            atom_mask=out["final_atom_mask"],
            aatype=final_aatype,
            residue_index=final_residue_index,
            b_factors=plddt_b_factors,
            chain_index=final_chain_index,
        )
    else:
        num_res = final_aatype.shape[0]
        chain_index = np.zeros(num_res, dtype=np.int32)
        return protein.Protein(
            atom_positions=out["final_atom_positions"],
            atom_mask=out["final_atom_mask"],
            aatype=final_aatype,
            residue_index=final_residue_index,
            b_factors=plddt_b_factors,
            chain_index=chain_index,
        )

def relax_protein(
    config,
    model_device,
    unrelaxed_protein,
    output_directory,
    output_name,
    cif_output=False,
):
    amber_relaxer = relax.AmberRelaxation(
        use_gpu=(model_device != "cpu"),
        **config.relax,
    )

    t = time.perf_counter()
    visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", default="")
    if "cuda" in model_device:
        device_no = model_device.split(":")[-1]
        os.environ["CUDA_VISIBLE_DEVICES"] = device_no
    # the struct_str will contain either a PDB-format or a ModelCIF format string
    struct_str, _, _ = amber_relaxer.process(
        prot=unrelaxed_protein, cif_output=cif_output
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
    relaxation_time = time.perf_counter() - t

    logger.info(f"Relaxation time: {relaxation_time}")
    update_timings(
        {"relaxation": relaxation_time}, os.path.join(output_directory, "timings.json")
    )

    # Save the relaxed PDB.
    suffix = "_relaxed.pdb"
    if cif_output:
        suffix = "_relaxed.cif"
    relaxed_output_path = os.path.join(output_directory, f"{output_name}{suffix}")
    with open(relaxed_output_path, "w") as fp:
        fp.write(struct_str)

    logger.info(f"Relaxed output written to {relaxed_output_path}...")
