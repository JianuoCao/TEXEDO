import os
import sys
# Make the repo root importable when run from generator/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import torch
import torch.nn.functional as F
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import subprocess
import time
import random
import shutil
import argparse
from mgpt.models.build_model import build_model
from mgpt.data.build_data import build_data
from mgpt.data.humanml.utils.word_vectorizer import WordVectorizer
# NOTE: upstream imported `convert_to_csv_format` from the (unpackaged) top-level
# `scripts/convert_output_to_36dim.py`, whose module body had hardcoded absolute
# paths and an eager legacy-VQVAE import. `mgpt.archs.fsq_arch` ships an
# identical `convert_to_csv_format` helper with none of that baggage.
from mgpt.archs.fsq_arch import convert_to_csv_format
from textseedo.paths import assets

# Run on GPU when available, else CPU (keeps the demo runnable without a GPU).
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def root_pos_to_vel_np(motion, root_dims=3):
    result = motion.copy()
    root_vel = np.zeros_like(motion[:, :root_dims])
    root_vel[1:] = motion[1:, :root_dims] - motion[:-1, :root_dims]
    if len(root_vel) > 1:
        root_vel[0] = root_vel[1]
    result[:, :root_dims] = root_vel
    return result


def get_matching_evaluator_params(cfg):
    """Read evaluator settings from the active MotionGPT config."""
    params = {
        't2m_path': str(assets("verifiers", "semantic", "t2m_custom36_combinedv2")),
        'max_motion_length': 2048,
        'unit_len': 4,
        'root_dims': 3,
    }

    try:
        params['t2m_path'] = cfg.METRIC.TM2T.t2m_path
    except (AttributeError, KeyError):
        pass

    for dataset_key in (
            'CUSTOM_COMBINED', 'CUSTOM_ALL', 'CUSTOM_LONG', 'CUSTOM',
            'HUMANML3D'):
        try:
            dataset_cfg = getattr(cfg.DATASET, dataset_key)
        except (AttributeError, KeyError):
            continue
        try:
            params['max_motion_length'] = int(dataset_cfg.MAX_MOTION_LEN)
        except (AttributeError, KeyError, TypeError, ValueError):
            pass
        try:
            params['unit_len'] = int(dataset_cfg.UNIT_LEN)
        except (AttributeError, KeyError, TypeError, ValueError):
            pass
        break

    return params

# # FSQ (default)
# python demo_ours.py --task t2m --num_samples 5

# # FSQ with explicit config + checkpoint
# python demo_ours.py --task t2m \
#     --cfg configs/config_custom_all_stage2_fsq.yaml \
#     --checkpoint experiments/mgpt/CustomDataAll_Stage2_FSQ/checkpoints/last.ckpt
# python demo_ours.py --task m2t \
#     --cfg configs/config_custom_all_stage2_fsq.yaml \
#     --checkpoint experiments/mgpt/CustomDataAll_Stage2_FSQ_m2t/checkpoints/last.ckpt
# # VQVAE (original)
# python demo_ours.py --task t2m \
#     --cfg configs/config_custom_all_stage2.yaml \
#     --checkpoint experiments/mgpt/CustomDataAll_Stage2_t2m/checkpoints/last.ckpt
# =============================================================================
# Evaluator for per-sample matching score
# =============================================================================
class MatchingScoreEvaluator:
    """Loads the 36-dim evaluator and computes per-sample matching scores."""

    def __init__(self,
                 device=DEVICE,
                 t2m_path=None,
                 max_motion_length=2048,
                 unit_len=4,
                 root_dims=3):
        from mgpt.archs.tm2t_evaluator import (
            TextEncoderBiGRUCo, MovementConvEncoder, MotionEncoderBiGRUCo
        )
        if t2m_path is None:
            t2m_path = assets("verifiers", "semantic", "t2m_custom36_combinedv2")
        self.device = device
        self.t2m_path = str(t2m_path)
        self.max_motion_length = int(max_motion_length)
        self.unit_len = int(unit_len)
        self.root_dims = int(root_dims)

        # Paths
        eval_ckpt = os.path.join(
            self.t2m_path, 'custom36', 't2m', 'text_mot_match', 'model',
            'finest.tar')
        eval_mean_path = os.path.join(
            self.t2m_path, 'custom36', 't2m', 'Comp_v6_KLD01', 'meta',
            'mean.npy')
        eval_std_path = os.path.join(
            self.t2m_path, 'custom36', 't2m', 'Comp_v6_KLD01', 'meta',
            'std.npy')

        # Load mean/std for z-normalizing raw motion before evaluator
        self.eval_mean = np.load(eval_mean_path)  # [36]
        self.eval_std = np.load(eval_std_path)     # [36]

        # Build evaluator networks
        self.text_enc = TextEncoderBiGRUCo(
            word_size=300, pos_size=15, hidden_size=512, output_size=512)
        self.move_enc = MovementConvEncoder(
            input_size=36, hidden_size=512, output_size=512)
        self.motion_enc = MotionEncoderBiGRUCo(
            input_size=512, hidden_size=1024, output_size=512)

        # Load weights
        ckpt = torch.load(eval_ckpt, map_location='cpu')
        self.text_enc.load_state_dict(ckpt['text_encoder'])
        self.move_enc.load_state_dict(ckpt['movement_encoder'])
        self.motion_enc.load_state_dict(ckpt['motion_encoder'])

        self.text_enc.eval().to(device)
        self.move_enc.eval().to(device)
        self.motion_enc.eval().to(device)

        # Use the real WordVectorizer (same as test mode)
        self.w_vectorizer = WordVectorizer(str(assets("glove")), 'our_vab')
        self.max_text_len = 50
        print(f"✓ Matching score evaluator loaded from {eval_ckpt}")

    def _encode_text(self, text_tokens):
        """Encode a tokenized text string into a 512-dim embedding.
        text_tokens: list of 'word/POS' strings.
        """
        if len(text_tokens) < self.max_text_len:
            tokens = ['sos/OTHER'] + text_tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens += ['unk/OTHER'] * (self.max_text_len + 2 - sent_len)
        else:
            tokens = text_tokens[:self.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)

        word_embs = []
        pos_ohs = []
        for tok in tokens:
            we, po = self.w_vectorizer[tok]
            word_embs.append(we[None, :])
            pos_ohs.append(po[None, :])

        word_embs = torch.tensor(np.concatenate(word_embs, 0), dtype=torch.float32).unsqueeze(0).to(self.device)  # [1, T, 300]
        pos_ohs = torch.tensor(np.concatenate(pos_ohs, 0), dtype=torch.float32).unsqueeze(0).to(self.device)      # [1, T, 15]
        cap_lens = torch.tensor([sent_len], dtype=torch.long).to(self.device)

        with torch.no_grad():
            text_emb = self.text_enc(word_embs, pos_ohs, cap_lens)  # [1, 512]
        return text_emb

    def _encode_motion(self, motion_36dim):
        """Encode raw 36-dim motion with the same padding contract as TM2TMetrics36Dim."""
        motion_36dim = np.asarray(motion_36dim, dtype=np.float32)
        if len(motion_36dim) == 0:
            return None

        valid_len = min(len(motion_36dim), self.max_motion_length)
        motion_36dim = motion_36dim[:valid_len]

        motion_36dim = root_pos_to_vel_np(
            motion_36dim, root_dims=self.root_dims)
        motion = (motion_36dim - self.eval_mean) / self.eval_std

        if valid_len < self.max_motion_length:
            pad = np.zeros(
                (self.max_motion_length - valid_len, motion.shape[1]),
                dtype=motion.dtype)
            motion = np.concatenate([motion, pad], axis=0)

        motion_t = torch.tensor(
            motion, dtype=torch.float32).unsqueeze(0).to(self.device)
        m_len = torch.tensor(
            [max(valid_len // self.unit_len, 1)],
            dtype=torch.long,
            device=self.device)

        with torch.no_grad():
            mov = self.move_enc(motion_t).detach()        # [1, T/4, 512]
            emb = self.motion_enc(mov, m_len).detach()    # [1, 512]
        return emb

    def compute_matching_score(self, text_tokens, motion_36dim):
        """Compute Euclidean distance between text and motion embeddings.
        Lower = better alignment.
        Returns float or None if motion is too short.
        """
        text_emb = self._encode_text(text_tokens)
        motion_emb = self._encode_motion(motion_36dim)
        if motion_emb is None:
            return None
        dist = F.pairwise_distance(text_emb, motion_emb).item()
        return dist

def demo_ours(task="t2m", num_samples=3, temperature=0.9, top_k=50, top_p=0.95,
              cfg_path="./configs/config_fsq_multitask.yaml",
              cfg_assets_path="./configs/assets.yaml",
              checkpoint_path=None, example_path=None, out_dir=None,
              exact_out_dir=False, batch_size=16, skip_existing=False,
              index_offset=0):
    # Load config
    print(f"Loading config from {cfg_path}")
    import sys
    sys.argv = ['demo_ours.py', '--cfg', cfg_path, '--cfg_assets', cfg_assets_path]
    from mgpt.config import parse_args
    cfg = parse_args(phase="test")
    print(f"Config loaded")
    print(f"   Dataset: {cfg.DATASET.target}")

    # 2. Build dataset
    datamodule = build_data(cfg)
    datamodule.setup()
    # 3. Build model
    model = build_model(cfg, datamodule)
    
    # Load checkpoint
    if checkpoint_path is None:
        # Default to the released generator checkpoint named by the config
        # (TEST.CHECKPOINTS -> ${TSD_ASSETS}/generator/epoch=489.ckpt).
        checkpoint_path = cfg.TEST.CHECKPOINTS
        print(f"Checkpoint path not specified, using config default: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found!")
        return
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval().to(DEVICE)

    # Set sampling parameters for diverse generation
    model.lm.set_sampling_params(temperature=temperature, top_k=top_k, top_p=top_p)
    print(f"Sampling params: temperature={temperature}, top_k={top_k}, top_p={top_p}")
    # setup output directory
    if exact_out_dir and out_dir:
        output_dir = Path(out_dir)
    else:
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        output_root = out_dir or cfg.FOLDER
        output_dir = Path(
            os.path.join(output_root, str(cfg.model.target.split('.')[-2]), str(cfg.NAME),
                         f"samples_{task}_" + timestamp))
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")
    print(f"Task: {task}")
    
    # Load matching score evaluator for t2m
    evaluator = None
    if task == 't2m' and example_path is None:
        try:
            evaluator = MatchingScoreEvaluator(
                device=DEVICE, **get_matching_evaluator_params(cfg))
        except Exception as e:
            print(f"WARNING: Could not load evaluator: {e}")
            print("Matching scores will not be computed.")
    
    # Read text inputs
    if example_path is None:
        # Load train.txt and test.txt
        train_file = os.path.join(cfg.DATASET.DATAPATH, 'train.txt')
        test_file = os.path.join(cfg.DATASET.DATAPATH, 'test.txt')
        texts_dir = os.path.join(cfg.DATASET.DATAPATH, 'texts')
        motion_dir = os.path.join(cfg.DATASET.DATAPATH, 'new_joint_vecs')
        
        # Read indices from train.txt and test.txt
        with open(train_file, 'r') as f:
            train_indices = [line.strip() for line in f.readlines() if line.strip()]
        with open(test_file, 'r') as f:
            test_indices = [line.strip() for line in f.readlines() if line.strip()]
        
        # Randomly sample 10 from each
        selected_train = random.sample(train_indices, min(2, len(train_indices)))
        # selected_test = random.sample(test_indices, min(10, len(test_indices)))
        # selected_indices = selected_train + selected_test
        selected_indices = selected_train
        print(f"Selected {len(selected_indices)} samples: {len(selected_train)} from train, 0 from test")
        
        # Process each selected index
        for motion_id in tqdm(selected_indices, desc="Generating"):
            # Read text file and randomly select one line
            text_file = os.path.join(texts_dir, f"{motion_id}.txt")
            if not os.path.exists(text_file):
                print(f"Text file not found: {text_file}")
                continue
            
            with open(text_file, 'r') as f:
                text_lines = [line.split('#')[0].strip() for line in f.readlines() if line.strip()]
            
            if not text_lines:
                print(f"No text found in {text_file}")
                continue
            
            text = random.choice(text_lines)
            
            # Load ground truth motion for m2t and m2m tasks
            gt_motion_file = os.path.join(motion_dir, f"{motion_id}.npy")
            if not os.path.exists(gt_motion_file):
                print(f"Ground truth not found: {gt_motion_file}")
                continue
            gt_motion = np.load(gt_motion_file)  # (seq_len, 36)
            
            # Create subdirectory for this motion
            motion_output_dir = output_dir / motion_id
            motion_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Task-specific generation
            if task == "t2m":
                # Text-to-Motion: generate motion from text
                # Save input text and ground truth (only once)
                with open(motion_output_dir / 'in.txt', 'w') as f:
                    f.write(text)
                gt_csv = convert_to_csv_format(gt_motion)
                np.savetxt(str(motion_output_dir / 'gt.csv'), gt_csv, delimiter=',')
                
                # Parse text tokens for matching score
                text_file_path = os.path.join(texts_dir, f"{motion_id}.txt")
                text_tokens = None
                with open(text_file_path, 'r') as f:
                    for line in f.readlines():
                        parts = line.strip().split('#')
                        if len(parts) >= 2 and parts[0].strip() == text:
                            text_tokens = parts[1].strip().split(' ')
                            break
                if text_tokens is None:
                    # Fallback: use first line's tokens
                    with open(text_file_path, 'r') as f:
                        first_line = f.readline().strip().split('#')
                        if len(first_line) >= 2:
                            text_tokens = first_line[1].strip().split(' ')
                        else:
                            text_tokens = [w + '/OTHER' for w in text.lower().split()]
                
                # Compute GT matching score
                scores_dict = {'text': text, 'motion_id': motion_id, 'samples': {}}
                if evaluator is not None:
                    gt_score = evaluator.compute_matching_score(text_tokens, gt_motion)
                    scores_dict['gt_matching_score'] = gt_score
                    print(f"  GT matching score for {motion_id}: {gt_score:.4f}")
                
                # Generate multiple samples
                for sample_idx in range(num_samples):
                    with torch.no_grad():
                        output_ids = model.lm.generate_conditional(
                            texts=[text],
                            task="t2m",
                            stage='test',
                            tasks=None
                        )
                    
                    # Decode tokens to motion
                    if output_ids and len(output_ids) > 0:
                        motion_tokens = output_ids[0]
                        motion_ids = [token_id for token_id in motion_tokens.tolist() 
                                      if 0 <= token_id < model.lm.m_codebook_size]
                        
                        if len(motion_ids) > 0:
                            # Decode tokens → 36-dim via model.vae (works for VQVAE and FSQ)
                            raw_tokens = torch.tensor(
                                motion_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
                            with torch.no_grad():
                                gen_36dim = model.vae.decode(
                                    raw_tokens).squeeze(0).cpu().numpy()  # [T, 36]
                            
                            # Save CSV
                            csv_path = str(motion_output_dir / f'out{sample_idx+1}.csv')
                            converted = convert_to_csv_format(gen_36dim)
                            np.savetxt(csv_path, converted, delimiter=',')
                            print(f"  Generated sample {sample_idx+1}/{num_samples} for {motion_id}: {len(motion_ids)} tokens")
                            
                            # Compute matching score using the 36-dim directly
                            if evaluator is not None:
                                sample_score = evaluator.compute_matching_score(text_tokens, gen_36dim)
                                scores_dict['samples'][f'out{sample_idx+1}'] = {
                                    'matching_score': sample_score,
                                    'num_tokens': len(motion_ids),
                                    'num_frames': len(gen_36dim),
                                }
                                if sample_score is not None:
                                    print(f"    Matching score: {sample_score:.4f}")
                
                # Save scores JSON
                if evaluator is not None:
                    json_path = motion_output_dir / 'matching_scores.json'
                    with open(json_path, 'w') as f:
                        json.dump(scores_dict, f, indent=2)
                    print(f"  Scores saved to {json_path}")
            
            elif task == "m2t":
                # Motion-to-Text: generate text from motion
                # Encode ground truth motion to tokens (following val_m2t_forward)
                motion_tensor = torch.from_numpy(gt_motion).float().unsqueeze(0).to(DEVICE)  # (1, seq, 36)
                
                with torch.no_grad():
                    # Unified encode: returns (tokens [1,T'], lengths [1,]) for both VQVAE and FSQ
                    motion_tokens_enc, _ = model.vae.encode(motion_tensor)
                    motion_tokens = [motion_tokens_enc[0]]
                    lengths_tokens = [motion_tokens_enc.shape[1]]
                
                # Save input motion and ground truth text (only once)
                in_csv = convert_to_csv_format(gt_motion)
                np.savetxt(str(motion_output_dir / 'in.csv'), in_csv, delimiter=',')
                with open(motion_output_dir / 'gt.txt', 'w') as f:
                    f.write(text)
                
                # Generate multiple text descriptions
                for sample_idx in range(num_samples):
                    with torch.no_grad():
                        output_ids = model.lm.generate_conditional(
                            motion_tokens=motion_tokens,
                            lengths=lengths_tokens,
                            task="m2t",
                            stage='test'
                        )
                    
                    # Decode tokens to text
                    if output_ids and len(output_ids) > 0:
                        generated_text = output_ids[0] if isinstance(output_ids[0], str) else str(output_ids[0])
                        with open(motion_output_dir / f'out{sample_idx+1}.txt', 'w') as f:
                            f.write(generated_text)
                        print(f"  Generated text {sample_idx+1}/{num_samples} for {motion_id}: {generated_text[:50]}...")
            
            elif task == "pred":
                # Motion-to-Motion Prediction: predict future motion
                
                # Encode input motion to tokens (following val_m2m_forward)
                motion_tensor = torch.from_numpy(gt_motion).float().unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    # Unified encode: works for both VQVAE and FSQ
                    motion_tokens_enc, _ = model.vae.encode(motion_tensor)
                    motion_tokens = [motion_tokens_enc[0]]
                    lengths_tokens = [motion_tokens_enc.shape[1]]
                gt_csv = convert_to_csv_format(gt_motion)
                np.savetxt(str(motion_output_dir / 'gt.csv'), gt_csv, delimiter=',')
                with open(motion_output_dir / 'info.txt', 'w') as f:
                    f.write(f"Description: {text}\n")
                
                # Generate multiple predictions
                for sample_idx in range(num_samples):
                    with torch.no_grad():
                        output_ids = model.lm.generate_conditional(
                            motion_tokens=motion_tokens,
                            lengths=lengths_tokens,
                            task="pred",
                            stage='test'
                        )
                    
                    # Decode tokens to motion
                    if output_ids and len(output_ids) > 0:
                        motion_tokens_out = output_ids[0]
                        motion_ids = [token_id for token_id in motion_tokens_out.tolist() 
                                      if 0 <= token_id < model.lm.m_codebook_size]
                        
                        if len(motion_ids) > 0:
                            raw_tokens = torch.tensor(
                                motion_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
                            with torch.no_grad():
                                pred_36dim = model.vae.decode(
                                    raw_tokens).squeeze(0).cpu().numpy()
                            csv_path = str(motion_output_dir / f'out{sample_idx+1}.csv')
                            np.savetxt(csv_path, convert_to_csv_format(pred_36dim), delimiter=',')
                            print(f"  Generated prediction {sample_idx+1}/{num_samples} for {motion_id}: {len(motion_ids)} tokens → {len(pred_36dim)} frames")
        
        print(f"Saved {len(selected_indices)} motions to {output_dir}")
    
    else:
        # Original logic: read from DEMO.EXAMPLE file
        with open(example_path, 'r') as f:
            texts = [line.strip() for line in f.readlines() if line.strip()]
        
        skipped_existing = 0
        generated_outputs = 0

        for b in tqdm(range((len(texts) + batch_size - 1) // batch_size), desc="Generating"):
            batch_texts = texts[b * batch_size:(b + 1) * batch_size]

            for sample_idx in range(num_samples):
                suffix = f'out{sample_idx + 1}' if num_samples > 1 else 'out'
                pending = []
                for i, text in enumerate(batch_texts):
                    idx = index_offset + b * batch_size + i
                    csv_path = output_dir / f'{idx}_{suffix}.csv'
                    txt_path = output_dir / f'{idx}_{suffix}.txt'
                    if skip_existing and csv_path.exists() and txt_path.exists():
                        skipped_existing += 1
                        continue
                    pending.append((idx, text))

                if not pending:
                    continue

                # Generate one sample for every text in this batch.
                with torch.no_grad():
                    output_ids = model.lm.generate_conditional(
                        texts=[text for _, text in pending],
                        task="t2m",
                        stage='test',
                        tasks=None
                    )

                # Decode and save each motion separately because token lengths vary.
                for i, (idx, text) in enumerate(pending):
                    if output_ids and i < len(output_ids):
                        motion_tokens = output_ids[i]
                        # Remove special tokens and extract motion IDs
                        motion_ids = []
                        for token_id in motion_tokens.tolist():
                            if 0 <= token_id < model.lm.m_codebook_size:
                                motion_ids.append(token_id)

                        if len(motion_ids) > 0:
                            raw_tokens = torch.tensor(
                                motion_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
                            with torch.no_grad():
                                gen_36dim = model.vae.decode(
                                    raw_tokens).squeeze(0).cpu().numpy()
                            csv_path = str(output_dir / f'{idx}_{suffix}.csv')
                            np.savetxt(csv_path, convert_to_csv_format(gen_36dim), delimiter=',')
                            # Also save the raw (T, 36) motion for the scoring pipeline.
                            np.save(str(output_dir / f'{idx}_{suffix}.npy'), gen_36dim.astype(np.float32))
                            with open(output_dir / f'{idx}_{suffix}.txt', 'w') as f:
                                f.write(text)
                            generated_outputs += 1
        
        print(f"Saved {len(texts)} motions to {output_dir}")
        if skip_existing:
            print(f"Skipped existing outputs: {skipped_existing}")
            print(f"Generated missing outputs: {generated_outputs}")
    
    print(f"\nTo visualize, run:")
    print(f"  python scripts/visualize_npz.py --input-dir {output_dir} --output-dir {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate motion samples from text')
    parser.add_argument('--cfg', type=str,
                       default='./configs/config_fsq_multitask.yaml',
                       help='Config file path (default: config_fsq_multitask.yaml)')
    parser.add_argument('--cfg_assets', type=str,
                       default='./configs/assets.yaml',
                       help='Asset config file path (default: assets.yaml)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Model checkpoint path. Defaults to experiments/<NAME>/checkpoints/last.ckpt')
    parser.add_argument('--example', type=str, default=None,
                       help='Text file with one prompt per line for batch t2m generation')
    parser.add_argument('--out_dir', type=str, default=None,
                       help='Root directory for generated samples (default: cfg.FOLDER)')
    parser.add_argument('--exact_out_dir', action='store_true',
                       help='Use --out_dir exactly instead of appending model/name/timestamp.')
    parser.add_argument('--task', type=str, default='t2m',
                       choices=['t2m', 'm2t', 'pred'],
                       help='Task type: t2m (text-to-motion), m2t (motion-to-text), pred (motion prediction)')
    parser.add_argument('--num_samples', type=int, default=3,
                       help='Number of samples to generate per input (default: 3)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for prompt-file generation (default: 16)')
    parser.add_argument('--temperature', type=float, default=0.9,
                       help='Sampling temperature (higher=more diverse, default: 0.9)')
    parser.add_argument('--top_k', type=int, default=50,
                       help='Top-k sampling (0=disabled, default: 50)')
    parser.add_argument('--top_p', type=float, default=0.95,
                       help='Nucleus (top-p) sampling (1.0=disabled, default: 0.95)')
    parser.add_argument('--skip_existing', action='store_true',
                       help='Skip prompt/sample outputs whose CSV and TXT already exist')
    parser.add_argument('--index_offset', type=int, default=0,
                       help='Global index offset for output filenames when sharding prompts across GPUs')
    args = parser.parse_args()
    
    demo_ours(task=args.task, num_samples=args.num_samples,
              temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
              cfg_path=args.cfg, cfg_assets_path=args.cfg_assets,
              checkpoint_path=args.checkpoint,
              example_path=args.example, out_dir=args.out_dir,
              exact_out_dir=args.exact_out_dir, batch_size=args.batch_size,
              skip_existing=args.skip_existing,
              index_offset=args.index_offset)
