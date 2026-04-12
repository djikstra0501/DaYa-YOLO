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
from ultralytics.utils import LOGGER

# =========================================================
# HELPER: PARSE FREEZE LAYERS
# =========================================================
def _parse_freeze_layers(freeze_input):
    """
    Parses custom freeze inputs into a list of integers.
    Supports: "1-11", "20", "2,6,8", or integers.
    """
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
    """Handles the Aux Loss weight scheduling for PGI-Style models."""
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
        crit._last_aux_epoch = epoch
        
    except Exception as e:
        LOGGER.warning(f"[Aux Schedule Error] {e}")


# =========================================================
# FEATURE 2: DYNAMIC LAYER FREEZING
# =========================================================
def _handle_dynamic_freezing(trainer):
    """Handles freezing and unfreezing specific layers dynamically by epoch."""
    try:
        args = trainer.args
        freeze_input = getattr(args, "freeze_layers", None)
        freeze_epochs = getattr(args, "freeze_epochs", None)

        target_layers = _parse_freeze_layers(freeze_input)
        
        # If no layers specified, do nothing
        if not target_layers:
            return

        epoch = trainer.epoch
        model_seq = getattr(trainer.model, "model", None) # The nn.Sequential architecture
        if model_seq is None:
            return

        def _set_grad(requires_grad):
            """Sets gradients and returns a list of layer names for logging."""
            impacted_names = []
            for idx, layer in enumerate(model_seq):
                if idx in target_layers:
                    # Capture the module name (e.g., Conv, C3k2, SPPF)
                    m_name = layer.__class__.__name__
                    impacted_names.append(f"{idx}:{m_name}")
                    
                    for param in layer.parameters():
                        param.requires_grad = requires_grad
            return impacted_names

        # Case A: Start of training -> Freeze them
        if epoch == 0:
            layer_info = _set_grad(requires_grad=False)
            unfreeze_msg = f" until epoch {freeze_epochs}" if freeze_epochs else " indefinitely"
            
            # Format: 0:Identity, 1:Conv, 2:Conv...
            formatted_layers = ", ".join(layer_info)
            LOGGER.info(f"⭐ [Dynamic Freeze] Successfully frozen: [{formatted_layers}]{unfreeze_msg}.")

        # Case B: Reached the unfreeze epoch -> Unfreeze them
        elif freeze_epochs is not None and epoch == int(freeze_epochs):
            layer_info = _set_grad(requires_grad=True)
            formatted_layers = ", ".join(layer_info)
            LOGGER.info(f"🔥 [Dynamic Freeze] Unfrozen: [{formatted_layers}] at epoch {epoch}. Backbone is now learning!")

    except Exception as e:
        LOGGER.warning(f"[Dynamic Freeze Error] {e}")
