import torch
from ultralytics import YOLO
import thop

def verify_gflops_discrepancy(model_path, model_name):
    """Investigate why GFLOPs differ"""
    model = YOLO(model_path)
    
    print(f"\n🔍 Analyzing {model_name}: {model_path}")
    print("=" * 50)
    
    # Method 1: THOP (standard)
    dummy_input = torch.randn(1, 3, 640, 640)
    flops_thop, params_thop = thop.profile(model.model, inputs=(dummy_input,), verbose=False)
    gflops_thop = flops_thop / 1e9
    
    # Method 2: Ultralytics
    ultralytics_info = model.info()
    if isinstance(ultralytics_info, tuple):
        gflops_ultra = ultralytics_info[3]  # GFLOPs is typically 4th element
        params_ultra = ultralytics_info[1]
        layers_ultra = ultralytics_info[0]
    else:
        gflops_ultra = ultralytics_info.get('GFLOPs', 'N/A')
        params_ultra = ultralytics_info.get('params', 'N/A')
    
    print(f"THOP Calculation:")
    print(f"  - Parameters: {params_thop:,} ({params_thop/1e6:.2f}M)")
    print(f"  - GFLOPs: {gflops_thop:.2f}")
    
    print(f"Ultralytics Calculation:")
    print(f"  - Parameters: {params_ultra:,} ({params_ultra/1e6:.2f}M)")
    print(f"  - GFLOPs: {gflops_ultra:.2f}")
    
    print(f"Discrepancy Analysis:")
    print(f"  - Parameter difference: {abs(params_thop - params_ultra):,} params")
    print(f"  - GFLOPs ratio: {gflops_ultra/gflops_thop:.2f}x")
    
    return gflops_thop, gflops_ultra

# Test both models
print("=== GFLOPs DISCREPANCY ANALYSIS ===")
gflops_thop_base, gflops_ultra_base = verify_gflops_discrepancy('yolo11n.pt', 'Base Model')
gflops_thop_yours, gflops_ultra_yours = verify_gflops_discrepancy('yolov11-eca-sam-s3.pt', 'Your Model')

print(f"\n🎯 ARCHITECTURE CHANGE IMPACT (using THOP - pure model):")
thop_reduction = ((gflops_thop_base - gflops_thop_yours) / gflops_thop_base) * 100
print(f"GFLOPs reduced by: {thop_reduction:.1f}%")

print(f"\n🎯 ARCHITECTURE CHANGE IMPACT (using Ultralytics - full pipeline):")
ultra_reduction = ((gflops_ultra_base - gflops_ultra_yours) / gflops_ultra_base) * 100
print(f"GFLOPs reduced by: {ultra_reduction:.1f}%")