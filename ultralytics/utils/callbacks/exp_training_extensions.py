"""
YOLO training extensions and custom control logic.

This module extends the default Ultralytics YOLO training pipeline with
additional functionality that is not supported out-of-the-box. It includes:

    1. Architecture-specific arguments
       - Custom parameters required for modified model designs
         (e.g., auxiliary heads, attention modules)

    2. Advanced training controls
       - Selective layer freezing (non-sequential / pattern-based)
       - Epoch-based freeze scheduling (freeze/unfreeze over time)

    3. Training hooks and callbacks
       - Runtime logic for applying custom behaviors during training

Design Notes:
    - This module mixes argument handling and runtime behavior intentionally
      for flexibility, but may be refactored into separate components:
        * argument definitions
        * scheduling logic
        * execution hooks

    - Some features rely on internal Ultralytics trainer structure and may
      break if upstream APIs change.

Typical usage:
    Imported and registered before calling `model.train()` to inject
    custom behavior into the training lifecycle.

Author:
    djikstra0501

Last Updated:
    2026-04-12
"""

import math
import torch

# =========================================================
# HELPER: STATUS ANNOUNCER (LOGIC ONLY)
# =========================================================
def _get_training_status_msg(trainer):
    """Returns a summary string of active features."""
    args = trainer.args
    model = getattr(trainer, "model", None)
    
    freeze_input = getattr(args, "freeze_layers", None)
    freeze_until = getattr(args, "freeze_epochs", None)
    has_freeze = freeze_input not in (None, "None", "none", False, "False")
    
    has_aux = False
    if model:
        crit = getattr(model, "criterion", None)
        if crit and hasattr(crit, "aux_weight"):
            has_aux = True

    header = "═"*50
    if not has_freeze and not has_aux:
        return f"\n{header}\nℹ️  [Standard Mode] No custom extensions active.\n{header}\n"
    
    freeze_status = f"ON | Layers: {freeze_input} | Duration: until epoch {freeze_until}" if has_freeze else "OFF"
    aux_status = "ON (DualDDetectLoss detected)" if has_aux else "OFF"
    
    msg = [
        f"\n{header}",
        "🚀 [DaYa Extensions] Custom training logic active:",
        f"   > Dynamic Freezing: {freeze_status}",
        f"   > Aux Scheduling:   {aux_status}",
        f"{header}\n"
    ]
    return "\n".join(msg)

# =========================================================
# FEATURE 1: DUAL DETECT AUX LOSS SCHEDULER
# =========================================================
def _update_aux_loss_schedule(trainer):
    """Handles the Aux Loss weight scheduling."""
    try:
        model = getattr(trainer, "model", None)
        if model is None: return
        crit = getattr(model, "criterion", None)
        if crit is None or not hasattr(crit, "aux_weight"): return

        epoch = trainer.epoch
        args = trainer.args
        total_epochs = int(getattr(args, "epochs", 0))
        
        schedule = getattr(args, "aux_schedule", "linear")
        aux_start = float(getattr(args, "aux", 0.5))
        aux_end = float(getattr(args, "aux_end", 0.0))
        start_epoch = int(getattr(args, "aux_start_epoch", 0))
        end_epoch = int(getattr(args, "aux_end_epoch", 0))
        
        if end_epoch <= 0: end_epoch = max(total_epochs - 1, 0)
        if end_epoch < start_epoch: end_epoch = start_epoch

        if schedule in (None, "none", "constant"):
            aux = aux_start
        elif epoch <= start_epoch:
            aux = aux_start
        elif epoch >= end_epoch:
            aux = aux_end
        else:
            t = (epoch - start_epoch) / max(end_epoch - start_epoch, 1)
            if schedule == "cosine":
                aux = aux_end + 0.5 * (aux_start - aux_end) * (1.0 + math.cos(math.pi * t))
            else:
                aux = aux_start + (aux_end - aux_start) * t

        crit.aux_weight = aux
        setattr(args, "aux_current", aux)
    except Exception as e:
        return None

# =========================================================
# HELPER: WEIGHT INTEGRITY CHECK
# =========================================================
def _get_weight_dna(trainer, target_layers):
    """Calculates the sum of weights for the first frozen layer to prove zero learning."""
    try:
        if not target_layers:
            return ""
        
        model_seq = getattr(trainer.model, "model", None)
        # We check the first layer in your freeze list (usually Layer 1)
        sentinel_idx = target_layers[0] 
        layer = model_seq[sentinel_idx]
        
        # Get the sum of the first parameter tensor (usually weights)
        weight_sum = 0
        for param in layer.parameters():
            weight_sum += param.sum().item()
            break # Just checking the first tensor is enough for a 'DNA' proof
            
        return f" | Layer {sentinel_idx} DNA: {weight_sum:.10f}"
    except:
        return ""

# =========================================================
# FEATURE 2: DYNAMIC LAYER FREEZING (WITH INTEGRITY CHECK)
# =========================================================
def _execute_dynamic_freezing(trainer):
    """Handles math and returns status + weight DNA."""
    try:
        args = trainer.args
        freeze_input = getattr(args, "freeze_layers", None)
        freeze_epochs = getattr(args, "freeze_epochs", None)
        epoch = trainer.epoch
        
        target_layers = []
        if freeze_input and freeze_input not in (None, "None", "none", "False", False):
            if isinstance(freeze_input, int): target_layers = [freeze_input]
            else:
                parts = str(freeze_input).replace(" ", "").split(",")
                for p in parts:
                    if "-" in p:
                        s, e = map(int, p.split("-"))
                        target_layers.extend(range(s, e + 1))
                    elif p.isdigit(): target_layers.append(int(p))

        if not target_layers: return None

        model_seq = getattr(trainer.model, "model", None)
        if model_seq is None: return None

        # 1. Force state based on current epoch
        freeze_limit = int(freeze_epochs) if freeze_epochs else 999999
        should_be_frozen = epoch < freeze_limit

        for idx, layer in enumerate(model_seq):
            if idx in target_layers:
                for param in layer.parameters():
                    param.requires_grad = not should_be_frozen

        # 2. Generate the "Proof" DNA
        dna_str = _get_weight_dna(trainer, target_layers)

        # 3. Handle messaging
        if not hasattr(trainer, "_last_freeze_state"):
            trainer._last_freeze_state = None

        if should_be_frozen:
            state_msg = "STILL FROZEN"
            icon = "❄️"
            if trainer._last_freeze_state != "frozen":
                state_msg = "APPLIED FREEZE"
                icon = "⭐"
                trainer._last_freeze_state = "frozen"
            return f"{icon} [Dynamic Freeze] {state_msg} (Epoch {epoch} < {freeze_limit}){dna_str}"
        
        elif not should_be_frozen and trainer._last_freeze_state == "frozen":
            trainer._last_freeze_state = "unfrozen"
            return f"🔥 [Dynamic Freeze] LIMIT REACHED: Layers unfrozen at epoch {epoch}.{dna_str}"

        return None
    except Exception as e:
        return f"⚠️ [Dynamic Freeze Error] {e}"