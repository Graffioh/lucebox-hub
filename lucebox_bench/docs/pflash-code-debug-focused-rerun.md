# PFlash quality on focused InfiniteBench Code-Debug cases

This run compares full prefill with PFlash on Qwen3.6-27B at about 120K input tokens. The cohort includes six cases selected from an earlier run and five unseen cases. It is an enriched hard-case cohort, not an unbiased InfiniteBench score.

The run started on August 29, 2026, and finished on August 30, 2026.

## Results

| Case | Cohort role | Expected | Full prefill | PFlash | Outcome | PFlash effective tokens | Target-prefill speedup | TTFT speedup |
| ---: | --- | :---: | :---: | :---: | --- | ---: | ---: | ---: |
| 3 | Repeated shared failure | A | B, wrong | B, wrong | Both wrong | 36,522 | 6.07x | 1.15x |
| 4 | Repeated PFlash loss | C | C, correct | B, wrong | PFlash loss | 20,080 | 12.79x | 1.24x |
| 5 | Repeated PFlash loss | C | C, correct | D, wrong | PFlash loss | 19,292 | 13.34x | 1.24x |
| 7 | Repeated shared failure | B | D, wrong | C, wrong | Both wrong | 20,375 | 12.49x | 1.23x |
| 8 | Repeated PFlash loss | D | D, correct | B, wrong | PFlash loss | 24,279 | 10.22x | 1.18x |
| 11 | Repeated PFlash loss | B | B, correct | C, wrong | PFlash loss | 16,546 | 15.96x | 1.20x |
| 12 | Unseen | A | A, correct | C, wrong | New PFlash loss | 16,533 | 15.98x | 1.20x |
| 13 | Unseen | A | D, wrong | D, wrong | New shared failure | 16,644 | 15.82x | 1.20x |
| 14 | Unseen | B | B, correct | B, correct | Both correct | 16,621 | 15.85x | 1.20x |
| 15 | Unseen | B | B, correct | B, correct | Both correct | 18,879 | 13.66x | 1.21x |
| 16 | Unseen | D | D, correct | D, correct | Both correct | 18,367 | 14.11x | 1.22x |

Full prefill answered 8 of 11 cases correctly. PFlash answered 3 of 11 correctly and retained 3 of the 8 correct full-prefill answers. PFlash lost five cases that full prefill answered correctly and did not recover any full-prefill failure.

PFlash reduced aggregate target-prefill time by 12.30x. Aggregate end-to-end TTFT improved by 1.21x because PFlash spent a median 265.41 seconds on work before target prefill. The median effective prompt was 18,879 tokens, or 15.7% of the source prompt. All 11 effective prompts exceeded the nominal 16,384-token budget.

Both policies completed all requests without request errors or non-finite diagnostics.

## Repeat stability

Full-prefill predictions repeated exactly on all six selected cases. PFlash predictions, output lengths, finish reasons, and retained-token counts repeated exactly on cases 3, 4, 5, 7, and 8.

Case 11 remained wrong but did not repeat at the answer level. The earlier PFlash run predicted A, stopped normally, and generated 8,241 tokens. This run predicted C and reached the 16,384-token output limit. Its effective prompt changed from 16,538 to 16,546 tokens.

Cases 4, 5, and 8 are deterministic PFlash-only failures in these two runs. Cases 3 and 7 are deterministic shared failures. Case 12 is a new PFlash-only failure and needs one confirmation run before use as a deterministic fixture. Case 11 is useful as a PFlash stability stress case, not as an exact-answer fixture.

## Run configuration

| Setting | Value |
| --- | --- |
| Lucebox revision | `4bdf0c8fb32edeeb767305d9c82a08b39d45a2cb`, equal to `origin/main` at run time |
| Target model | `Qwen3.6-27B-Q4_K_M.gguf` |
| PFlash drafter | `Qwen3-0.6B-BF16.gguf` |
| Full-prefill input budget | 120,000 tokens |
| PFlash budget | 16,384 tokens |
| Query window | 128 tokens |
| Maximum output | 16,384 tokens |
| Temperature | 0 |
| Thinking | Enabled |
| Target device | `hip:0` |
| KV cache | `q4_0` K and V |
| Prefix cache slots | 0 |
| Flash-attention window | 0 |

Artifact hashes:

- Workload: `60b79665e5bdb41ce33f69dbf4876ea055747b32423dd207e4c7b1fdc7ecfba3`
- Lucebox binary: `524d1dad09b5cc3d04dfdf9a2ab2834276e02b5efbfd74e7d10170ab66696170`
- Target model: `5ed60d0af4650a854b1755bd392f9aef4872643dc25a254bc68043fa638392a0`
- PFlash drafter: `f9c9f1d3c1e21755b82d4e165f88dbbbd4355646d632fb5d6cef7c66ed4ee04e`

## Machine health

The guard recorded 1,560 samples and no violations. The peak ROCm junction temperature was 89 C, and peak benchmark-GPU VRAM use was 29.9 GiB. Available host memory stayed above 118.1 GiB, swap use stayed at zero, and free disk space stayed above 94.6 GiB.

After the run, the benchmark GPU reported 0% use, 0% allocated VRAM, and a 52 C junction temperature. No KFD processes remained.
