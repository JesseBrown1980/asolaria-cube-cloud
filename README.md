# asolaria-cube-cloud

Cloud-run surface for the Asolaria unified-Omega cube trainer (seat ACER-CLAUDE-FABLE5).
Runs OFF the operator's local machine — on cloud agents / GitHub Actions runners.

- `unified_omega.rs` — dependency-free Rust trainer: 3 families (8 cube-poles × 800 +
  12 sectors × 1200 + 20 Pi-lenses × 2000 = 40 bodies / 60,800 cells) from ONE shared
  0-loss reversible root; every cube its own SHA-seeded 1024-glyph language; byte-exact
  reversible every cell; two-level OmniSubmit fold to `omega_unified`. `claims_final_apex=0`.
- `scripts/<name>/corpus.bin` — each writing system's real Unicode sign inventory
  (assigned code points, UCD 15.1.0), tagged MEASURED_REAL_INVENTORY. NOT texts, no
  frequencies, no meaning — inventory only.
- `build.sh <name>` — compile + run one corpus, print the Omega hashes.

Deterministic: same corpus → byte-identical omega. Reversibility = lossless recall, NOT
a compression record.
