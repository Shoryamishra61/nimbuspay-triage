# Research-Driven Extension — Evidence, Hypotheses, Ablations, Paper Standard

## 1. Honest status

The supplied assessment is an applied ML fine-tuning task, not inherently an A* conference contribution. A single fictional support-ticket dataset with 1,445 noisy train, 200 dev and 400 hidden test examples—plus 30 exact train/dev overlaps—is insufficient by itself to claim a new general result. A paper-grade project needs a clear novelty claim, stronger uncontaminated evaluation, multiple seeds/models/domains, careful ablations, baselines, statistical uncertainty, reproducibility, license clearance and independent review.

**Assignment track:** prioritize compliant submission, no new inference machinery, no additional model-generated labels/predictions.

**Later research track:** may use additional public datasets and models if licensed and evaluated separately, but never retroactively present these as assignment test results.

## 2. Literature map — verified publisher/proceedings sources

| Reference | Venue and link | Relevant principle | What it does **not** prove here |
|---|---|---|---|
| Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* | ICLR 2022, https://openreview.net/pdf?id=nZeVKeeFYf9 | Frozen base + low-rank trainable adaptation | Does not validate chosen rank or expected score |
| Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs* | NeurIPS 2023, https://papers.neurips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html | NF4 and memory-efficient PEFT | Does not guarantee parity on this ticket task |
| Northcutt et al., *Confident Learning: Estimating Uncertainty in Dataset Labels* | JAIR 2021, https://research.google/pubs/confident-learning-estimating-uncertainty-in-dataset-labels/ | Data-label error detection and prudent pruning | Probabilistic label correction requires careful fit to task and restrictions |
| Liu et al., *Lost in the Middle* | TACL 2024, https://aclanthology.org/2024.tacl-1.9/ | Long-context relevant-text position can hurt accuracy | Does not tell us actual failure rate on quoted ticket history |
| Xie et al., *DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining* | NeurIPS 2023, https://papers.neurips.cc/paper_files/paper/2023/hash/dcba6be91359358c2355cd920da3fcbd-Abstract-Conference.html | Data-mixture selection matters | Full group-DRO method may be compute-wasteful here |
| Tirumala et al., *D4: Document De-Duplication and Diversification* | NeurIPS 2023, https://papers.neurips.cc/paper_files/paper/2023/hash/a8f8cbd7f7a5fb2c837e578c75e5b615-Abstract-Datasets_and_Benchmarks.html | Redundant examples and diversity matter | Pretraining-scale findings may not transfer to 1,445-row SFT |
| Lin et al., *Not All Tokens Are What You Need for Pretraining (Rho-1)* | NeurIPS 2024, https://proceedings.neurips.cc/paper_files/paper/2024/hash/3322a9a72a1707de14badd5e552ff466-Abstract-Conference.html | Informative-token selection can aid efficiency | More complex selective-loss research is not required to satisfy assistant-only masking |
| Agarwal et al., *DELIFT: Data Efficient Language Model Instruction Fine-Tuning* | ICLR 2025, https://proceedings.iclr.cc/paper_files/paper/2025/hash/f9d446812a6fdc05453f4093e54831e8-Abstract-Conference.html | Diversity/informativeness-aware SFT sample selection | Additional expensive selection can lose to simple methods at T4 budget |
| Yin and Rush, *Compute-Constrained Data Selection* | ICLR 2025, https://proceedings.iclr.cc/paper_files/paper/2025/hash/5acb720a361eecb34ee62d356859d246-Abstract-Conference.html | Selection compute must be counted against training gains | A fancy selector is not automatically compute-optimal |
| Fu et al., *T-SHIRT: Token-Selective Hierarchical Data Selection for Instruction Tuning* | NeurIPS 2025, https://proceedings.neurips.cc/paper_files/paper/2025/hash/a59ff5f7384176b5d14a5ded77c4aa4f-Abstract-Conference.html | Hierarchical/token-sensitive selection and robustness | No direct justification for copying an expensive method into this tiny task |
| Ma et al., *Task-Specific Data Selection via Monosemantic Neuronal Activations* | NeurIPS 2025, https://proceedings.neurips.cc/paper_files/paper/2025/hash/c971ffd786876d5927caf482b06bcba5-Abstract-Conference.html | Hidden-state/activation probes may inform selection | Sparse autoencoder probes likely exceed this assignment's budget |
| Yoo et al., *Code-Switching Curriculum Learning for Multilingual Transfer* | Findings ACL 2025, https://aclanthology.org/2025.findings-acl.407/ | Curriculum and code switching can affect transfer | Korean-focused results do not establish Hinglish gain |
| Bhattacharya et al., *Linguistic and Mathematical Reasoning in Multilingual Number Puzzles* | EMNLP 2025, https://aclanthology.org/2025.emnlp-main.1438/ | Language-number interactions are brittle | Must validate rupee/k/lakh arithmetic directly |
| Moon et al., *Call for Rigor in Reporting Quality of Instruction Tuning Data* | ACL 2025, https://aclanthology.org/2025.acl-short.9/ | Hyperparameter confounding can invalidate data-quality conclusions | Does not tell us which data treatment wins |
| Geirhos et al., *Shortcut Learning in Deep Neural Networks* | Nature Machine Intelligence 2020, https://www.nature.com/articles/s42256-020-00257-z | Shortcuts and spurious correlations undermine robustness | Requires counterfactual tests for this problem |

### Recommended books (foundational, not a substitute for empirical evidence)

- Chip Huyen, *Designing Machine Learning Systems*, O'Reilly 2022: https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/ — lifecycle reliability, data, monitoring and experimental choices.
- Simon J. D. Prince, *Understanding Deep Learning*, MIT Press 2023: https://mitpress.mit.edu/9780262377102/understanding-deep-learning/ — representation/optimization fundamentals.
- Goodfellow, Bengio and Courville, *Deep Learning*, MIT Press 2016: https://www.deeplearningbook.org/ — optimization, generalization and experimental methodology.

## 3. Research question and falsifiable hypotheses

**Proposed research theme:** *Invariant-aware curation and counterfactual training for reliable structured extraction with small quantized language models under a fixed-prompt inference constraint.* This is a **candidate direction**, not a demonstrated novel method.

**RQ1:** Does a train-only invariant-based audit produce higher exact-match generalization than raw or simple syntactic cleaning under equal training compute?

- H1a: More correct labels improve exact match and P1 rule consistency.
- H1b: Over-aggressive pruning degrades rare Hinglish and long-context performance.

**RQ2:** Does data diversity / targeted boundary sampling help numerical and conditional compositional generalization more than uniform sampling at fixed token/step budget?

- H2a: Boundary-aware sampling improves ₹5k/₹50k transitions on uncontaminated holdouts.
- H2b: Gains may not transfer to novel date and ID formats; measure separately.

**RQ3:** How much accuracy/memory/time is lost or gained by QLoRA vs FP16 LoRA at fixed model and adapter rank?

- H3: 4-bit quantization reduces memory; task accuracy parity is empirical and can fail.

**RQ4:** Are mistakes dominated by input ambiguity, training label noise, weak numerical reasoning, long-position distraction, or output-format violations?

- H4: Multiple failure mechanisms interact; raw exact match alone hides them.

**RQ5 (later only):** Does invariant-driven curation transfer across multiple structured extraction domains/models without test-time rule scaffolding?

## 4. Research method and ablation matrix

### Submission-legal minimal study

Reference `cleaned, 4-bit QLoRA, rank8, uniform`; controlled variants `raw`, `FP16 LoRA`, `rank16`, `hard-case sampling` with the same base, seed, optimizer-step budget, prompt, loss masking and generation rules. Two mandatory untouched base-model dev baselines reported separately. Never tune on hidden test.

### Post-assignment expansion for a credible submission to peer review

1. Collect multiple independent licensed datasets/domains, with truly held-out *group-level* splits to prevent paraphrase/template leakage.
2. Compare raw, syntax-only cleaning, invariant-only cleaning, manual-verified cleaning, confidence filtering, random-size-matched pruning, and proposed method.
3. Evaluate at least three model sizes/families and >=3 seeds per key condition, consistent compute/tokens and frozen selection protocol.
4. Quantify confidence intervals (bootstrap over independent examples/groups), paired differences, effect sizes and multiple-comparison handling.
5. Use targeted counterfactual stress sets designed *before* measuring winner: threshold ±1 paise, year rollover, distractor amount placement, legal phrases, code-switch variants, long quoted histories, false NP IDs.
6. Measure exact match, field accuracy, schema validity, constraint violation, critical error rates, compute tokens/seconds, peak memory, data quality audit precision/recall on a small human-validated set.
7. Perform ablations isolating each audit layer, pruning type, weight schedule, augmentation family, rank, context length, adapter precision, and budget.
8. Test statistical robustness, external validity, fairness across language groups, reproducible code/data release permission, and negative results.
9. Write a detailed ethics and limitations section; don't claim real-world fraud safety from a toy dataset.
10. Establish novelty against DELIFT, T-SHIRT, Confident Learning, Rho-1, shortcut-learning and data-cleaning baselines; a mere combination of known methods is insufficient by itself.

## 5. Prohibited shortcuts even in 'research' framing during the assignment

- Cannot use GPT/Claude/another LLM as a label generator, judge of training labels, test prediction source or ensemble voter.
- Cannot insert schema rules into final inference or repair generated JSON.
- Cannot train on the 200 dev examples or 400 test inputs, including pseudo-labels.
- Cannot use dev labels to create counterfactual train labels or optimize dozens of hypotheses without acknowledging selection bias.
- Cannot invent metrics, p-values, ablations, model runs, or claimed state-of-the-art performance.

## 6. Publication decision gate

**Paper-ready? NO**, not until experiments have been completed and reviewed, generalization is shown on independent domains, a real novelty claim survives prior-art comparison, uncertainty is quantified, and artifacts can legally be released. Targeting NeurIPS/ICLR/ACL is a motivation for rigor, not a guarantee of acceptance or a quality label that can be asserted ahead of peer review.

