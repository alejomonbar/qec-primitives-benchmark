# Presentations

- `qec_primitives.html` — "Evaluating QEC primitives on quantum processors, at large width and depth". One
  self-contained page (only the fonts come from Google Fonts, with system fonts as the fallback offline).
  - Title page: live d = 5 surface-code patches running syndrome extraction (CNOT layers, mid-circuit measurement,
    detection events); hover a check for its last outcomes, a data qubit for the checks that watch it.
  - Part 1, steps 1–7: syndrome extraction on a d = 3 surface code, following `qecbench.codes.surface_code`,
    `memory.check_types` and `memory.cx_schedule`: the code and its checks, one Z check, one X check, a full round
    in four CNOT layers, mid-circuit measurement and the syndrome record, errors / detection events / decoding, and
    feed-forward.
  - Part 2, step 8: how many shots a surface-code memory experiment needs. ε_L per round comes from stim +
    pymatching simulations of `memory.stim_circuit` (uniform circuit-level noise p, R = d rounds, Z basis) for
    d = 3, 5, 7, fitted as ε_L ≈ 0.0257 (p / 0.0129)^((d+1)/2) (within 8 % for p ≤ 5×10⁻³; d ≥ 9 extrapolated).
    Shots for ±10 % on ε_L: S ≈ 100 / (R ε_L), one basis. Hover the chart to change p (default 10⁻³).
  - Part 3, steps 9–11: our proposal, syndrome extraction as LR-QAOA on the XZ Hamiltonian
    (`circuits.build_dynamic(frame="XZ")`). Step 9: the measurement of one check and the LR-QAOA gadget differ by one
    gate (H → Rx(2γ)) plus feed-forward. Step 10: one layer on the whole d = 3 patch (Z checks, X checks, Rx, MCM,
    feed-forward, Y mixer) has the CZs and MCMs of one syndrome round. Step 11: r_ovl against depth on ibm_phoenix
    (d = 3 run 20261005_071721, 11 patches; d = 5 run 20261002_121845, 8 patches; 1,000 shots per circuit).
  - Part 4, steps 12–14: across QPUs and codes. Step 12: ibm_phoenix against ibm_chicago, d = 3, each patch's XZ
    r_ovl at p = 2 against the logical error per round of the memory experiment of the same day (phoenix
    20261005_071721 / 20261005_071730, 11 patches; chicago 20261111_135143 / 20261111_141428, 9 patches);
    Spearman ρ 0.94 and 0.87, 0.94 pooled. Step 13: surface, triangular colour and bivariate-bicycle codes on
    Helios-1 hardware (every check as Z…Z), r against depth, noiseless r where it can be simulated.
    Step 14: paper Fig. 6, r at the best depth against N_q for the three families on Helios-1, H2-1,
    ibm_phoenix and the emulators (runs listed in the step's note).
  - Part 5, steps 15–18: the mid-circuit gadget itself. Step 15: how λ is measured (paper Fig. 5a): the depolarizing
    simulation of N_q = 5–15 chains collapses onto r_ovl = 2^(−κ_r ε_acc), κ_r = κ₀/N_q, κ₀ = 3.69, with ibm_boston
    N_q = 10 as an example. Step 16: r at p = 3 of every data–ancilla–data triplet per device as violins (paper Fig. 3c
    plus phoenix 20260916_102941 and chicago 20261111_123350 / 20261117_101043). Next to them, the coupling maps of ibm_boston
    and ibm_chicago with each qubit coloured by r of the triplet it is the ancilla of. Step 17: λ_eff against chain length,
    N_q = 10–60, MCM (paper Fig. 5c plus chicago 20261111_120600, best and median placement). Step 18: λ_eff per chip
    against its first public availability (paper Fig. 5d), chicago drawn hollow at the date of the run.
  - → / Space / click: next; ←: back (from the start of step 1 to the title page).
  - F: full-screen presentation mode (a clicker's PageDown / PageUp work); Esc exits. Full screen shows the heading, the drawing and the step text side by side; C hides the step text.
  - `?t=<seconds>` opens at that time without the title page; `?cover=0` skips the title page; `?present=1` opens in the full-screen layout (for screenshots).
