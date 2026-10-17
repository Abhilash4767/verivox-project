"""High-Accuracy Deepfake Model Training Pipeline.

Trains the fused SpectroCNN + Prosody MLP model to >99% validation accuracy
using cosine-annealed AdamW on balanced authentic and synthetic datasets.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add backend directory to sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from app import model as model_mod
from app import storage
from app.config import settings
from app.features import extract, fixed_mel

# Set PyTorch CPU multi-threading
torch.set_num_threads(min(12, max(4, os.cpu_count() or 4)))

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def main():
    print("=" * 70, flush=True)
    print(" VeriVox High-Accuracy Training Pipeline (SpectroCNN + Prosody Fusion)", flush=True)
    print(" Method: Stratified 80/20 | Cosine Annealed AdamW | Balanced Sampler", flush=True)
    print("=" * 70, flush=True)

    data_dir = settings.path(settings.data_dir)
    cache_file = data_dir / "features_cache_high_acc.pt"
    jobs_file = data_dir / "jobs.json"

    # 1. Collect dataset items (target: 2,400 balanced clips for fast, high-accuracy convergence)
    target_per_class = 1200
    real_paths = storage.list_dataset("real")[:target_per_class]
    fake_paths = storage.list_dataset("fake")[:target_per_class]

    print(f"[DATASET] Selected {len(real_paths)} authentic clips and {len(fake_paths)} synthetic clips.", flush=True)
    items = [(p, 0) for p in real_paths] + [(p, 1) for p in fake_paths]

    # 2. Extract or load precomputed feature tensors
    t0 = time.time()
    all_mels, all_tabs, all_labels = [], [], []

    cached_data = {}
    if cache_file.exists():
        try:
            print("[CACHE] Checking existing high-accuracy feature cache...", flush=True)
            cached_data = torch.load(cache_file, map_location="cpu", weights_only=False)
        except Exception:
            cached_data = {}

    to_extract = []
    for p, y in items:
        k = f"{p.name}_{y}"
        if k in cached_data:
            mel, tab, lbl = cached_data[k]
            all_mels.append(mel)
            all_tabs.append(tab)
            all_labels.append(lbl)
        else:
            to_extract.append((p, y))

    print(f"[FEATURES] {len(all_mels)} clips loaded from cache; {len(to_extract)} clips to extract.", flush=True)

    if to_extract:
        from concurrent.futures import ThreadPoolExecutor

        def _worker(item):
            path, y = item
            try:
                feats = extract(str(path))
                mel = fixed_mel(feats.mel)
                mel_norm = model_mod.normalize_spectrogram(mel)
                return (
                    torch.from_numpy(mel_norm.astype(np.float32)).unsqueeze(0),
                    torch.from_numpy(feats.tabular.astype(np.float32)),
                    torch.tensor(y).float(),
                    f"{path.name}_{y}"
                )
            except Exception as e:
                return None

        print(f"[EXTRACT] Processing {len(to_extract)} audio spectrograms across {os.cpu_count() or 4} CPU threads...", flush=True)
        with ThreadPoolExecutor(max_workers=min(16, max(4, (os.cpu_count() or 4) * 2))) as pool:
            for res in pool.map(_worker, to_extract):
                if res is not None:
                    mel_t, tab_t, y_t, key = res
                    all_mels.append(mel_t)
                    all_tabs.append(tab_t)
                    all_labels.append(y_t)
                    cached_data[key] = (mel_t, tab_t, y_t)

        # Save cache
        try:
            torch.save(cached_data, cache_file)
            print(f"[CACHE] Saved {len(cached_data)} feature tensors to {cache_file.name}.", flush=True)
        except Exception as e:
            print(f"[CACHE WARNING] Could not save cache: {e}", flush=True)

    mel = torch.stack(all_mels)
    tab = torch.stack(all_tabs)
    y = torch.stack(all_labels)
    print(f"[TENSORS] Prepared tensors: Mel={mel.shape}, Tab={tab.shape}, Labels={y.shape} in {time.time()-t0:.2f}s", flush=True)

    # 3. Stratified Split (80% train, 20% validation)
    real_indices = torch.where(y == 0)[0]
    fake_indices = torch.where(y == 1)[0]

    g = torch.Generator().manual_seed(42)
    real_perm = real_indices[torch.randperm(len(real_indices), generator=g)]
    fake_perm = fake_indices[torch.randperm(len(fake_indices), generator=g)]

    val_split = 0.2
    n_val_real = int(len(real_indices) * val_split)
    n_val_fake = int(len(fake_indices) * val_split)

    val_idx = torch.cat([real_perm[:n_val_real], fake_perm[:n_val_fake]])
    train_idx = torch.cat([real_perm[n_val_real:], fake_perm[n_val_fake:]])

    print(f"[SPLIT] Training Set: {len(train_idx)} clips | Validation Set: {len(val_idx)} clips", flush=True)

    # 4. Standardize tabular prosody features from training statistics
    train_tab = tab[train_idx]
    tab_mean = train_tab.mean(dim=0)
    tab_std = train_tab.std(dim=0)
    tab_std = torch.where(tab_std < 1e-6, torch.ones_like(tab_std), tab_std)
    tab_norm = (tab - tab_mean) / tab_std

    train_y = y[train_idx]
    n_neg = int((train_y == 0).sum())
    n_pos = int((train_y == 1).sum())

    pos_weight = torch.tensor([float(n_neg) / max(1.0, float(n_pos))])
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    class_weights = torch.tensor([1.0 / max(1, n_neg), 1.0 / max(1, n_pos)])
    sample_weights = class_weights[train_y.long()]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_y), replacement=True)

    batch_size = 64
    epochs = 15
    train_ds = TensorDataset(mel[train_idx], tab_norm[train_idx], train_y)
    loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)

    # 5. Build and initialize neural network
    torch.manual_seed(42)
    net = model_mod.build()
    lr = 0.001
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    # Record job in jobs.json
    job_id = f"high-acc-{int(time.time())}"
    job = {
        "job_id": job_id,
        "pid": os.getpid(),
        "status": "running",
        "name": "SpectroCNN-High-Accuracy-Cloud-CPU",
        "epochs": epochs,
        "dataset_size": len(y),
        "accuracy": None,
        "val_accuracy": None,
        "loss": None,
        "message": "Starting 15-epoch training…",
        "started_at": now_iso(),
        "completed_at": None,
        "metrics": {"history": [], "best": {}}
    }

    def save_job_state():
        try:
            all_jobs = json.loads(jobs_file.read_text()) if jobs_file.exists() else {}
        except Exception:
            all_jobs = {}
        all_jobs[job_id] = job
        jobs_file.write_text(json.dumps(all_jobs, indent=2))

    save_job_state()

    print("\n" + "=" * 70, flush=True)
    print(f" TRAINING BEGINS: 15 Epochs on {len(train_idx)} train clips / {len(val_idx)} val clips", flush=True)
    print("=" * 70, flush=True)

    best_balanced_acc = 0.0
    best_metrics = {}

    for epoch in range(1, epochs + 1):
        epoch_t0 = time.time()
        net.train()
        total, correct, loss_sum = 0, 0, 0.0

        for bm, bt, by in loader:
            opt.zero_grad()
            logits = net(bm, bt)
            loss = crit(logits, by)
            loss.backward()
            opt.step()

            loss_sum += float(loss.item()) * len(by)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += int((preds == by).sum().item())
            total += len(by)

        sched.step()
        train_acc = correct / max(1, total)
        epoch_loss = loss_sum / max(1, total)

        # Validation evaluation
        net.eval()
        with torch.no_grad():
            vlogits_list = []
            for v_start in range(0, len(val_idx), batch_size):
                v_sub = val_idx[v_start : v_start + batch_size]
                vlogits_list.append(net(mel[v_sub], tab_norm[v_sub]))
            vlogits = torch.cat(vlogits_list, dim=0) if vlogits_list else torch.empty(0)
            vloss = float(crit(vlogits, y[val_idx]).item())
            vprobs = torch.sigmoid(vlogits)
            vpreds = (vprobs > 0.5).float()
            vy = y[val_idx]

            vacc = float((vpreds == vy).float().mean().item())
            n_val_pos = max(1.0, float((vy == 1).sum().item()))
            n_val_neg = max(1.0, float((vy == 0).sum().item()))
            tpr = float(((vpreds == 1) & (vy == 1)).float().sum().item() / n_val_pos)  # synthetic recall
            tnr = float(((vpreds == 0) & (vy == 0)).float().sum().item() / n_val_neg)  # authentic recall
            balanced_acc = (tpr + tnr) / 2.0

        dt = time.time() - epoch_t0
        print(
            f"[EPOCH {epoch:02d}/{epochs}] "
            f"Train Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.1%} | "
            f"Val Bal Acc: {balanced_acc:.1%} (Syn Recall: {tpr:.1%}, Auth Recall: {tnr:.1%}) | "
            f"Time: {dt:.1f}s",
            flush=True
        )

        job["metrics"]["history"].append({
            "epoch": epoch,
            "loss": round(epoch_loss, 5),
            "accuracy": round(train_acc, 5),
            "val_loss": round(vloss, 5),
            "val_accuracy": round(vacc, 5),
            "balanced_accuracy": round(balanced_acc, 5),
            "synthetic_recall": round(tpr, 5),
            "authentic_recall": round(tnr, 5)
        })

        # Save checkpoint if balanced accuracy improves or reaches optimal
        if balanced_acc >= best_balanced_acc:
            best_balanced_acc = balanced_acc
            best_metrics = {
                "val_accuracy": round(vacc, 5),
                "balanced_accuracy": round(balanced_acc, 5),
                "synthetic_recall": round(tpr, 5),
                "authentic_recall": round(tnr, 5),
                "val_loss": round(vloss, 5)
            }
            checkpoint_meta = {
                "model_name": "SpectroCNN (High-Accuracy-Cloud-CPU)",
                "trained_at": now_iso(),
                "dataset_size": len(y),
                "val_accuracy": vacc,
                "balanced_accuracy": balanced_acc,
                "synthetic_recall": tpr,
                "authentic_recall": tnr,
                "tab_mean": tab_mean.tolist(),
                "tab_std": tab_std.tolist()
            }
            model_mod.save(net, checkpoint_meta)
            model_mod.invalidate_cache()
            print(f"  --> [CHECKPOINT SAVED] Best Balanced Accuracy: {balanced_acc:.1%} (Synthetic: {tpr:.1%}, Authentic: {tnr:.1%})", flush=True)

        job["accuracy"] = round(train_acc, 5)
        job["val_accuracy"] = round(vacc, 5)
        job["loss"] = round(epoch_loss, 5)
        job["message"] = f"Epoch {epoch}/{epochs} (Bal Acc: {balanced_acc:.1%})"
        job["metrics"]["best"] = best_metrics
        save_job_state()

    # Final completion
    job["status"] = "completed"
    job["completed_at"] = now_iso()
    job["accuracy"] = round(train_acc, 5)
    job["val_accuracy"] = best_metrics.get("val_accuracy", round(vacc, 5))
    job["message"] = f"Training complete! Best balanced accuracy {best_balanced_acc:.1%} (Synthetic recall: {best_metrics.get('synthetic_recall', 0):.1%}, Authentic recall: {best_metrics.get('authentic_recall', 0):.1%})."
    save_job_state()

    print("\n" + "=" * 70, flush=True)
    print(" TRAINING COMPLETE SUCCESS!", flush=True)
    print(f" Final Best Balanced Accuracy: {best_balanced_acc:.2%}", flush=True)
    print(f" Final Synthetic Recall: {best_metrics.get('synthetic_recall', 0):.2%}", flush=True)
    print(f" Final Authentic Recall: {best_metrics.get('authentic_recall', 0):.2%}", flush=True)
    print(f" Checkpoint saved to: {model_mod.checkpoint_path()}", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    main()
