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
from ultralytics.utils import LOGGER, RANK

# =========================================================
# HELPER: STATUS ANNOUNCER
# =========================================================
def _announce_training_status(trainer):
    """Prints a one-time summary of which custom features are active."""
    # Only shout on the main GPU
    if RANK not in (-1, 0):
        return

    args = trainer.args
    model = getattr(trainer, "model", None)
    
    # Check for Dynamic Freezing
    freeze_input = getattr(args, "freeze_layers", None)
    has_freeze = freeze_input not in (None, "None", "none", False, "False")
    
    # Check for Aux Loss Support (DualDDetect/PGI models only)
    has_aux = False
    if model:
        crit = getattr(model, "criterion", None)
        if crit and hasattr(crit, "aux_weight"):
            has_aux = True

    print("\n" + "="*50, flush=True)
    if not has_freeze and not has_aux:
        msg = "ℹ️  [Standard Mode] No custom extensions active. Running vanilla YOLO training."
        print(msg, flush=True)
        LOGGER.info(msg)
    else:
        msg = "🚀 [DaYa Extensions] Custom training logic active:"
        print(msg, flush=True)
        LOGGER.info(msg)
        
        freeze_status = f"ON (Layers: {freeze_input})" if has_freeze else "OFF"
        aux_status = "ON (DualDDetectLoss detected)" if has_aux else "OFF (Not a PGI model)"
        
        print(f"   > Dynamic Freezing: {freeze_status}", flush=True)
        print(f"   > Aux Scheduling:   {aux_status}", flush=True)
    
    print("="*50 + "\n", flush=True)


# =========================================================
# HELPER: PARSE FREEZE LAYERS
# =========================================================
def _parse_freeze_layers(freeze_input):
    if not freeze_input or freeze_input in (None, "None", "none", "False", False):
        return []
    if isinstance(freeze_input, int):
        return [freeze_input]
    layers = set()
    parts = str(freeze_input).replace(" ", "").split(",")
    for part in parts:
        if "-" in part:
            start, end = map(int, part.split("-"))
            layers.update(range(start, end + 1))
        elif part.isdigit():
            layers.add(int(part))
    return sorted(list(layers))


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
        if RANK in (-1, 0): LOGGER.warning(f"[Aux Schedule Error] {e}")


# =========================================================
# FEATURE 2: DYNAMIC LAYER FREEZING
# =========================================================
def _handle_dynamic_freezing(trainer):
    """Handles dynamic freezing/unfreezing."""
    try:
        args = trainer.args
        freeze_input = getattr(args, "freeze_layers", None)
        freeze_epochs = getattr(args, "freeze_epochs", None)

        target_layers = _parse_freeze_layers(freeze_input)
        if not target_layers: return

        epoch = trainer.epoch
        model_seq = getattr(trainer.model, "model", None)
        if model_seq is None: return

        def _set_grad(requires_grad):
            impacted_names = []
            for idx, layer in enumerate(model_seq):
                if idx in target_layers:
                    m_name = layer.__class__.__name__
                    impacted_names.append(f"{idx}:{m_name}")
                    for param in layer.parameters():
                        param.requires_grad = requires_grad
            return impacted_names

        is_main = RANK in (-1, 0)
        
        if epoch == 0:
            layer_info = _set_grad(requires_grad=False)
            if is_main:
                unfreeze_msg = f" until epoch {freeze_epochs}" if freeze_epochs else " indefinitely"
                msg = f"[Dynamic Freeze] Successfully frozen: [{', '.join(layer_info)}]{unfreeze_msg}."
                print(msg, flush=True)
                LOGGER.info(msg)

        elif freeze_epochs is not None and epoch == int(freeze_epochs):
            layer_info = _set_grad(requires_grad=True)
            if is_main:
                msg = f"[Dynamic Freeze] Unfrozen: [{', '.join(layer_info)}] at epoch {epoch}. Model active!"
                print(msg, flush=True)
                LOGGER.info(msg)

    except Exception as e:
        if RANK in (-1, 0): LOGGER.warning(f"[Dynamic Freeze Error] {e}")