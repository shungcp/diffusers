from huggingface_hub import hf_hub_download
import json
import os

# 1. Download Standard Config (Reference)
print("Downloading Standard Config...")
ref_path = hf_hub_download("hunyuanvideo-community/HunyuanVideo", "config.json", subfolder="transformer")
with open(ref_path, 'r') as f:
    ref_conf = json.load(f)

# 2. Download FP8 Config (Source)
print("Downloading FP8 Config...")
fp8_path = hf_hub_download("Aquiles-ai/HunyuanVideo-1.5-720p-fp8", "config.json", subfolder="transformer/720p_t2v")
with open(fp8_path, 'r') as f:
    fp8_conf = json.load(f)

# 3. Compare Keys
print("\n--- CONFIG COMPARISON ---")
ref_keys = set(ref_conf.keys())
fp8_keys = set(fp8_conf.keys())

common = ref_keys.intersection(fp8_keys)
missing = ref_keys - fp8_keys
extra = fp8_keys - ref_keys

print(f"Common Keys: {len(common)}")
print(f"Missing Keys (Present in Std, Missing in FP8): {len(missing)}")
for k in sorted(list(missing)):
    print(f"  - {k} (Std Value: {ref_conf[k]})")

print(f"\nExtra Keys (Present in FP8, Missing in Std): {len(extra)}")
for k in sorted(list(extra)):
    print(f"  - {k} (FP8 Value: {fp8_conf[k]})")

# 4. Generate Mapped Config
# We start with Ref config (to ensure all needed keys exist) and override with FP8 values where possible
print("\n--- GENERATING MAPPED CONFIG ---")
mapped_conf = ref_conf.copy()

# MAPPING LOGIC (Hypothesis based on names)
mapping = {
    # Std Key : FP8 Key
    "hidden_size": "hidden_size", 
    "num_attention_heads": "heads_num",
    "num_layers": "mm_double_blocks_depth", 
    "num_single_layers": "mm_single_blocks_depth",
    "patch_size": "patch_size", 
    "patch_size_t": "patch_size_t",
    "in_channels": "in_channels",
    "out_channels": "out_channels",
    # Mappings for mismatches
    "mlp_ratio": "mlp_width_ratio",
    "rope_axes_dim": "rope_dim_list",
    
    # Embedding Dims
    # Standard: text_embed_dim=4096. FP8: text_states_dim=3584 ? 
    # If 1.5 uses different text encoder dim, we must map it.
    "text_embed_dim": "text_states_dim",
    "pooled_projection_dim": "vision_states_dim", # Approx mapping, needed for verifying shapes
}

# Special handling for 720p specific params if needed
# We want to force the dimension to 2048
if "hidden_size" in fp8_conf:
    mapped_conf["hidden_size"] = fp8_conf["hidden_size"]
    print(f"Force updated hidden_size to: {mapped_conf['hidden_size']}")

# Generic Mapping Loop
print("\n--- APPLYING MAPPINGS ---")
for std_key, fp8_key in mapping.items():
    if fp8_key in fp8_conf:
        val = fp8_conf[fp8_key]
        
        # SANITIZATION: Ensure scalar integers for dimensions
        # If it's a list (e.g. [2] or [2, 2]), take the first element.
        if isinstance(val, list) and std_key in ["patch_size", "patch_size_t", "in_channels", "out_channels", "hidden_size", "num_attention_heads"]:
             print(f"  Sanitizing {std_key}: {val} -> {val[0]}")
             val = val[0]
             
        mapped_conf[std_key] = val
        print(f"Mapped {fp8_key} ({val}) -> {std_key}")
    else:
        print(f"Warning: FP8 key {fp8_key} not found for {std_key}")

# Manual overrides for HunyuanVideo 1.5 (Inferred from mismatch errors)
# MOVED AFTER LOOP to prevent overwriting
print("\n--- APPLYING 1.5 OVERRIDES (Post-Mapping) ---")
if "1.5" in str(fp8_path) or True: # Force for this debugging session
    mapped_conf["in_channels"] = 65
    mapped_conf["out_channels"] = 32 # Assuming patch_size 1*1*1 -> 32. Or patch_size 2*2*1 -> 8? 
    # Checkpoint kernel was 1x1x1 for x_embedder -> patch_size=(1,1,1).
    # If patch_size is 1, then proj_out dim = out_channels * 1 * 1 * 1 = 32. So out_channels=32.
    # Note: Ref config has patch_size=2. We must ensure patch_size is updated if it comes from FP8 conf.
    # If FP8 conf doesn't correct patch_size to 1, we must force it.
    
    # Let's inspect patch_size from fp8_conf in the loop below, but override if needed.
    mapped_conf["patch_size"] = 1
    mapped_conf["patch_size_t"] = 1
    
    print(f"Force updated in_channels to: 65")
    print(f"Force updated out_channels to: 32")
    print(f"Force updated patch_size to: 1")
    
    # 1472 is the correct dim for this checkpoint (ByT5 Pooled)
    mapped_conf["pooled_projection_dim"] = 1472
    print(f"Force updated pooled_projection_dim to: 1472")


# Manual Overrides for missing keys or specific fixes
if "attn_head_dim" not in mapped_conf and "hidden_size" in mapped_conf and "num_attention_heads" in mapped_conf:
    # Calculate head dim if missing
    mapped_conf["attention_head_dim"] = mapped_conf["hidden_size"] // mapped_conf["num_attention_heads"]
    print(f"Calculated attention_head_dim: {mapped_conf['attention_head_dim']}")


# Check for mismatches
print("\nResulting Mapped Config (Sample):")
print(f"hidden_size: {mapped_conf['hidden_size']}")
print(f"num_attention_heads: {mapped_conf['num_attention_heads']}")

# Save locally to use
output_path = "hunyuan_fp8_transformer_link/mapped_config.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(mapped_conf, f, indent=2)
print(f"\nSaved mapped config to: {output_path}")
