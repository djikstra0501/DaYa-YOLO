import torch
from ultralytics.nn.tasks import DetectionModel

def inspect_spectral_encoder(pt_path):
    ckpt = torch.load(pt_path, map_location='cpu', weights_only=False)
    state = ckpt['model'].state_dict()
    
    print(f"\n{'='*60}")
    print(f"Inspecting: {pt_path}")
    print(f"{'='*60}")
    
    # lab_scale or xyz_scale (the learned sliders)
    for key in state.keys():
        if 'lab_scale' in key or 'xyz_scale' in key:
            scale = state[key].squeeze()
            print(f"\n--- Learned Channel Scale ({key}) ---")
            print(f"  L* / X  weight: {scale[0].item():.6f}")
            print(f"  a* / Y  weight: {scale[1].item():.6f}")
            print(f"  b* / Z  weight: {scale[2].item():.6f}")
            
            dominant = scale.argmax().item()
            names = ['L*/X', 'a*/Y', 'b*/Z']
            print(f"\n  Dominant channel: {names[dominant]} ({scale[dominant].item():.6f})")
            print(f"  Ratio to others:  {scale[dominant].item() / scale.mean().item():.3f}x mean")

    # encoder weights
    for key in state.keys():
        if 'model.12.encoder.weight' in key:
            w = state[key].squeeze()
            print(f"\n--- Encoder 1x1 Conv Weights ({key}) ---")
            print(f"  Shape: {w.shape}  (out_channels x in_channels)")
            channel_names = ['L*/X', 'a*/Y', 'b*/Z']
            for i in range(w.shape[0]):
                row = w[i]
                dominant = row.abs().argmax().item()
                print(f"  Out ch {i}: [{row[0].item():+.4f}, {row[1].item():+.4f}, {row[2].item():+.4f}]"
                      f"  ← dominant input: {channel_names[dominant]}")

    print(f"\n{'='*60}\n")


models = {
    "LAB Color":  "propose_daya_color_lab.pt",
    "Strict XYZ": "propose_daya_color_xyz.pt",
}

for name, path in models.items():
    print(f"\n{'#'*60}")
    print(f"  MODEL: {name}")
    print(f"{'#'*60}")
    inspect_spectral_encoder(path)