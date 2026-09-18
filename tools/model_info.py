import torch
from ultralytics import YOLO
import thop
import pandas as pd

# === Define model paths ===
model_paths = {
    "Base (YOLO11n)": "yolo11n.pt",
    "Stage 1": "yolov11-eca-sam-s1.pt",
    "Stage 2": "yolov11-eca-sam-s2.pt",
    "Stage 3": "yolov11-eca-sam-s3.pt"
}

def simple_model_analysis(model_paths_dict):
    """Simple analysis using only THOP"""
    results = []
    
    for name, path in model_paths_dict.items():
        try:
            print(f"🔍 Analyzing {name}...")
            model = YOLO(path)

            # Dummy input (same as YOLO training size)
            dummy_input = torch.randn(1, 3, 640, 640)

            # Calculate FLOPs and parameters
            flops, params = thop.profile(model.model, inputs=(dummy_input,), verbose=False)

            gflops = flops / 1e9
            params_m = params / 1e6

            results.append({
                'Model': name,
                'Parameters': f"{params:,}",
                'Parameters_M': f"{params_m:.2f}M", 
                'GFLOPs': f"{gflops:.2f}",
                'Model_Path': path.split('/')[-1]
            })

            print(f"   ✅ {params_m:.2f}M parameters, {gflops:.2f} GFLOPs")

        except Exception as e:
            print(f"   ❌ Error analyzing {name}: {e}")
            results.append({
                'Model': name, 'Parameters': 'Error',
                'Parameters_M': 'Error', 'GFLOPs': 'Error',
                'Model_Path': path
            })

    return pd.DataFrame(results)

# === Run analysis ===
print("=== SIMPLE MODEL ANALYSIS ===")
df_simple = simple_model_analysis(model_paths)
print("\n" + "="*60)
print(df_simple.to_string(index=False))

# === Compare models with base ===
if len(df_simple) > 1 and all(df_simple['Parameters_M'] != 'Error'):
    print("\n=== ARCHITECTURE CHANGES ===")
    base_params = float(df_simple.iloc[0]['Parameters_M'].replace('M', ''))
    base_gflops = float(df_simple.iloc[0]['GFLOPs'])
    
    for i in range(1, len(df_simple)):
        current_params = float(df_simple.iloc[i]['Parameters_M'].replace('M', ''))
        current_gflops = float(df_simple.iloc[i]['GFLOPs'])
        
        param_change = ((current_params - base_params) / base_params) * 100
        gflops_change = ((current_gflops - base_gflops) / base_gflops) * 100
        
        print(f"{df_simple.iloc[0]['Model']} → {df_simple.iloc[i]['Model']}:")
        print(f"  Parameters: {base_params:.2f}M → {current_params:.2f}M ({param_change:+.1f}%)")
        print(f"  GFLOPs:     {base_gflops:.2f} → {current_gflops:.2f} ({gflops_change:+.1f}%)\n")
