"""Default env flags for the luce daemon when spawned by pflash."""

# These are the only daemon-side flags pflash assumes. The C++ kernel knobs
# (LUCE_FP_USE_BSA, LUCE_FP_ALPHA) are set per-call by the daemon owner.
LUCE_REQUIRED_ENV = {
    "LUCE_FA_WINDOW": "0",       # full attn on the (already compressed) prompt
    "LUCE_KV_TQ3": "1",          # 3-bit KV cache; saves ~4 GB at 128K
    "LUCE_LM_HEAD_FIX": "0",     # disable cuBLAS LM-head dequant (OOMs on 24 GB)
}
