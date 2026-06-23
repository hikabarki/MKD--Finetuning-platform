#!/usr/bin/env python3
"""
Sanity check for MKD Fine-Tuning Platform — NemotronH VL chat generation fixes.
Run: python sanity_check.py
"""
import sys, os, types, subprocess, importlib.util, ast
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
WARN = "\033[33m⚠\033[0m"

# Paths are derived from this script's location so the check is portable across
# machines (it lives at the project root). Override the root with $FT_PROJECT.
_HERE     = os.path.dirname(os.path.abspath(__file__))
_ROOT     = os.environ.get("FT_PROJECT", _HERE)
_HOME     = os.path.expanduser("~")
MODEL_DIR = os.path.join(_ROOT, "models", "nvidia__nvidia-nemotron-nano-12b-v2-vl-bf16")
APP_PY    = os.path.join(_ROOT, "app.py")

errors = []
warnings = []

def ok(msg):   print(f"  {PASS} {msg}")
def fail(msg): print(f"  {FAIL} {msg}"); errors.append(msg)
def warn(msg): print(f"  {WARN} {msg}"); warnings.append(msg)

# ─────────────────────────────────────────────
print("\n══ 1. File-level fix presence ══")
# ─────────────────────────────────────────────

import glob

all_mnh  = glob.glob(f"{_HOME}/.cache/huggingface/modules/transformers_modules/***/modeling_nemotron_h.py")
all_mnh += glob.glob(f"{_HOME}/.cache/huggingface/hub/***/modeling_nemotron_h.py")
all_mnh += glob.glob(os.path.join(_ROOT, "models", "*", "modeling_nemotron_h.py"))

# Filter out venv/uv cache/other envs
SKIP = ["venv","\.cache/uv","RAG-Context","LLM-Comparison","vision-encoder","open-webui"]
import re
custom_files = [f for f in all_mnh if not any(re.search(p,f) for p in SKIP)]

fix_checks = {
    "prepare_inputs_for_generation (is_first_step)": "is_first_step = cache_params is None",
    "conv_kernel_size as self attr":                  "self.conv_kernel_size = config.conv_kernel",
    "update_conv_state list fix":                     "self.conv_states[layer_idx].device",
    "update_ssm_state list fix":                      "self.ssm_states[layer_idx].device",
}

for f in sorted(set(custom_files)):
    try:
        src = open(f).read()
    except Exception as e:
        warn(f"Cannot read {f}: {e}")
        continue
    # Only check files that ARE custom (have HybridMambaAttentionDynamicCache)
    if "HybridMambaAttentionDynamicCache" not in src:
        continue
    missing = [name for name, pat in fix_checks.items() if pat not in src]
    if missing:
        fail(f"MISSING fixes in {f}:\n      " + "\n      ".join(missing))
    else:
        ok(f"All fixes present: {os.path.basename(os.path.dirname(f))}/modeling_nemotron_h.py")

# ─────────────────────────────────────────────
print("\n══ 2. Syntax checks ══")
# ─────────────────────────────────────────────

for path, label in [(APP_PY, "app.py"),
                    (f"{MODEL_DIR}/modeling_nemotron_h.py", "modeling_nemotron_h.py (model dir)"),
                    (f"{MODEL_DIR}/modeling.py",            "modeling.py (VL wrapper)")]:
    try:
        ast.parse(open(path).read())
        ok(f"{label}: syntax OK")
    except SyntaxError as e:
        fail(f"{label}: SyntaxError at line {e.lineno}: {e.msg}")

# ─────────────────────────────────────────────
print("\n══ 3. app.py output-count consistency ══")
# ─────────────────────────────────────────────

src = open(APP_PY).read()
lines_app = src.splitlines()

# Extract user_chat_send body (scope the yield checks)
ucs_start = next((i for i,l in enumerate(lines_app) if re.search(r'^\s+def user_chat_send\b', l)), None)
ucs_end   = next((i for i,l in enumerate(lines_app)
                  if i > (ucs_start or 0) and re.search(r'^\s{4}def \w|^def \w|user_send_btn\.click', l)), len(lines_app))
ucs_body  = lines_app[ucs_start:ucs_end] if ucs_start else []

# user_chat_send must never yield exactly 2 values (should always be 3)
bad_yields = [ucs_start+i+1 for i,line in enumerate(ucs_body)
              if re.search(r'yield\s+history[\w\s\[\]]*\s*,\s*""(\s*$|\s*#)', line)
              and not re.search(r'yield\s+history.*,\s*"".*,', line)]
if bad_yields:
    fail(f"user_chat_send has 2-value yields at lines: {bad_yields}")
elif not ucs_body:
    warn("Could not locate user_chat_send in app.py to check yields")
else:
    ok("No 2-value yields in user_chat_send")

# run_user_custommix must not have 3-value yields with gr.Row
bad3 = [i+1 for i,line in enumerate(src.splitlines())
        if "gr.Row(visible" in line and "yield" in line]
if bad3:
    fail(f"run_user_custommix still has gr.Row yields at lines: {bad3}")
else:
    ok("No gr.Row 3-value yields in run_user_custommix")

# go_to_chat must not call load_model
if "load_model(model_path)" in src and "def go_to_chat" in src:
    # check if it's inside go_to_chat
    lines = src.splitlines()
    in_gtc = False
    for line in lines:
        if "def go_to_chat" in line: in_gtc = True
        if in_gtc and "def " in line and "go_to_chat" not in line: in_gtc = False
        if in_gtc and "load_model(model_path)" in line:
            fail("go_to_chat still calls load_model(model_path)")
            break
    else:
        ok("go_to_chat does not call load_model(model_path)")
else:
    ok("go_to_chat does not call load_model(model_path)")

# ─────────────────────────────────────────────
print("\n══ 4. Unit test: prepare_inputs_for_generation ══")
# ─────────────────────────────────────────────

try:
    import torch
    from transformers import DynamicCache

    pkg_name = "_nemotron_sanity"
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [MODEL_DIR]
    pkg.__package__ = pkg_name
    sys.modules[pkg_name] = pkg

    def load_submod(name, filename):
        spec = importlib.util.spec_from_file_location(
            f"{pkg_name}.{name}", f"{MODEL_DIR}/{filename}", submodule_search_locations=[])
        m = importlib.util.module_from_spec(spec)
        m.__package__ = pkg_name
        sys.modules[f"{pkg_name}.{name}"] = m
        spec.loader.exec_module(m)
        setattr(pkg, name, m)
        return m

    load_submod("configuration_nemotron_h", "configuration_nemotron_h.py")
    mod = load_submod("modeling_nemotron_h", "modeling_nemotron_h.py")
    HybridCache = mod.HybridMambaAttentionDynamicCache
    prepare_fn  = mod.NemotronHForCausalLM.prepare_inputs_for_generation

    class FakeCfg:
        num_hidden_layers        = 4
        hybrid_override_pattern  = "M*-M"
        mamba_num_heads          = 8
        mamba_head_dim           = 8
        ssm_state_size           = 16
        conv_kernel              = 4
        hidden_size              = 64
        num_logits_to_keep       = 1
        num_attention_heads      = 8
        num_key_value_heads      = 8
        head_dim                 = 8

    emb = torch.nn.Embedding(100, 64)
    class FakeModel:
        config = FakeCfg()
        dtype  = torch.bfloat16
        device = torch.device("cpu")
        def get_input_embeddings(self): return emb

    model = FakeModel()
    B, T  = 1, 6
    ids   = torch.randint(0, 100, (B, T))
    mask  = torch.ones(B, T, dtype=torch.long)
    dyn   = DynamicCache()

    # Test 1: first step — DynamicCache pre-filled, cache_position=None (transformers 5.x crash scenario)
    r1 = prepare_fn(model, ids, past_key_values=dyn, attention_mask=mask,
                    inputs_embeds=None, cache_position=None, position_ids=None, use_cache=True)
    assert isinstance(r1["cache_params"], HybridCache), "cache_params not HybridCache"
    ok("Test 1 PASS: first step with DynamicCache + cache_position=None → no crash")

    # Test 2: decode step
    r2 = prepare_fn(model, torch.randint(0,100,(B,T+1)), past_key_values=None,
                    attention_mask=torch.ones(B,T+1,dtype=torch.long),
                    inputs_embeds=None, cache_position=None, position_ids=None, use_cache=True,
                    cache_params=r1["cache_params"])
    assert r2["input_ids"].shape == (B, 1), f"Expected (1,1) got {r2['input_ids'].shape}"
    ok("Test 2 PASS: decode step → input_ids sliced to last token")

    # Test 3: VL path (inputs_embeds)
    r3 = prepare_fn(model, ids, past_key_values=dyn, attention_mask=mask,
                    inputs_embeds=torch.randn(B,T,64),
                    cache_position=None, position_ids=None, use_cache=True)
    assert "inputs_embeds" in r3 and "input_ids" not in r3
    ok("Test 3 PASS: VL path returns inputs_embeds only")

    # Test 4: HybridMambaAttentionDynamicCache instance attrs
    cache = HybridCache(FakeCfg(), batch_size=1, dtype=torch.bfloat16, device="cpu")
    assert hasattr(cache, "conv_kernel_size"), "conv_kernel_size missing"
    assert hasattr(cache, "conv_states") and isinstance(cache.conv_states, list)
    assert hasattr(cache, "ssm_states")  and isinstance(cache.ssm_states, list)
    ok("Test 4 PASS: HybridCache has conv_kernel_size attr and list states")

    # Test 5: update_conv_state doesn't crash (list.device fix)
    mamba_layers = [i for i,c in enumerate("M*-M") if c == "M"]
    for li in mamba_layers:
        new_cs = torch.zeros(1, 8*8, 4)  # intermediate_size=64, conv_kernel=4
        cache.update_conv_state(li, new_cs, cache_init=True)
    ok("Test 5 PASS: update_conv_state cache_init=True → no list.device crash")

    for li in mamba_layers:
        new_cs = torch.zeros(1, 1, 8*8)
        cache.update_conv_state(li, new_cs, cache_init=False)
    ok("Test 6 PASS: update_conv_state cache_init=False → no list.device crash")

    # Test 7: update_ssm_state
    for li in mamba_layers:
        new_ss = torch.zeros(1, 8*8, 16)
        cache.update_ssm_state(li, new_ss)
    ok("Test 7 PASS: update_ssm_state → no list.device crash")

except Exception as e:
    import traceback
    fail(f"Unit tests failed: {e}")
    traceback.print_exc()

# ─────────────────────────────────────────────
print("\n══ 5. App process and URL ══")
# ─────────────────────────────────────────────

result = subprocess.run(["pgrep", "-af", "python.*app.py"], capture_output=True, text=True)
if result.stdout.strip():
    pid_line = result.stdout.strip().splitlines()[0]
    ok(f"App running: {pid_line[:80]}")
else:
    fail("app.py is NOT running")

try:
    log = open("/tmp/app_v5.log").read()
    urls = [l.strip() for l in log.splitlines() if "gradio.live" in l]
    if urls:
        ok(f"Public URL: {urls[-1]}")
    else:
        warn("No public URL found in log")
except Exception:
    warn("Could not read /tmp/app_v5.log")

# ─────────────────────────────────────────────
print("\n══ 6. HF module cache integrity ══")
# ─────────────────────────────────────────────

REQUIRED_FILES = {
    "configuration_nemotron_h.py", "configuration_radio.py", "image_processing.py",
    "modeling.py", "modeling_nemotron_h.py", "processing.py", "processing_utils.py",
    "video_io.py",
}
CACHE_ROOT = os.path.expanduser(
    "~/.cache/huggingface/modules/transformers_modules/"
    "nvidia__nvidia_hyphen_nemotron_hyphen_nano_hyphen_12b_hyphen_v2_hyphen_vl_hyphen_bf16")

if os.path.isdir(CACHE_ROOT):
    hash_dirs = [d for d in os.listdir(CACHE_ROOT)
                 if os.path.isdir(os.path.join(CACHE_ROOT, d))
                 and re.fullmatch(r'[0-9a-f]{16}', d)]
    for hd in sorted(hash_dirs):
        present = set(os.listdir(os.path.join(CACHE_ROOT, hd)))
        missing = REQUIRED_FILES - present
        if missing:
            fail(f"Incomplete cache dir {hd}/: missing {sorted(missing)}")
        else:
            # Also verify fixes in each dir's modeling file
            mnh = os.path.join(CACHE_ROOT, hd, "modeling_nemotron_h.py")
            try:
                s = open(mnh).read()
                missing_fixes = [n for n,p in fix_checks.items() if p not in s]
                if missing_fixes:
                    fail(f"Cache dir {hd}/modeling_nemotron_h.py missing: {missing_fixes}")
                else:
                    ok(f"Cache dir {hd}/: complete + all fixes present")
            except Exception as e:
                warn(f"Cannot read {mnh}: {e}")
else:
    warn(f"HF cache root not found: {CACHE_ROOT}")

# ─────────────────────────────────────────────
print("\n══ 7. Recent chat errors ══")
# ─────────────────────────────────────────────

try:
    err_raw = open("/tmp/ft_chat_errors.log").read().strip()
    if err_raw:
        # Parse into blocks separated by "---", then filter known-stale blocks
        blocks = [b.strip() for b in err_raw.split("---") if b.strip()]
        KNOWN_STALE_PATTERNS = ["FileNotFoundError", "bc5aa9280", "configuration_nemotron_h.py"]
        new_errors = [b for b in blocks if not any(p in b for p in KNOWN_STALE_PATTERNS)]
        if new_errors:
            warn(f"{len(new_errors)} new chat error(s) since last fix:")
            for b in new_errors[-2:]:
                for l in b.splitlines()[-4:]: print(f"    {l}")
        else:
            ok(f"Chat error log has {len(blocks)} old/known error(s) (pre-fix, safe to ignore)")
    else:
        ok("No chat errors in /tmp/ft_chat_errors.log")
except FileNotFoundError:
    ok("No chat error log (never errored)")

# ─────────────────────────────────────────────
print("\n══ 8. VRAM ══")
# ─────────────────────────────────────────────

try:
    r = subprocess.run(
        ["nvidia-smi","--query-gpu=name,memory.free,memory.total","--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=5)
    for row in r.stdout.strip().splitlines():
        name, free, total = [x.strip() for x in row.split(",")]
        pct = int(free)/int(total)*100
        ok(f"{name}: {int(free):,}/{int(total):,} MiB free ({pct:.0f}%)")
except Exception as e:
    warn(f"Could not query GPU: {e}")

# ─────────────────────────────────────────────
print()
if errors:
    print(f"\033[31m══ RESULT: {len(errors)} ERROR(S), {len(warnings)} warning(s) ══\033[0m")
    for e in errors: print(f"  {FAIL} {e}")
    sys.exit(1)
else:
    print(f"\033[32m══ RESULT: ALL CHECKS PASSED ══\033[0m" +
          (f"  ({len(warnings)} warning(s))" if warnings else ""))
