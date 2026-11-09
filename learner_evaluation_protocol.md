# Evaluation protocol for rated learner speech (fixed before the data were available)

Written on 2026-11-10, before the UME-JRF corpus was received. Every choice below is fixed; deviations will be reported as such.

## Frozen models

MoraCheck, trained on native speech and simulated errors with the uniform group mixture, checkpoint after the last step (12,000 steps):

| Run | SHA-256 of the checkpoint |
|---|---|
| moracheck_s0 | 1c6f9f433256fcd5b732e26d611472678eab0e9150c448a4f4f07b01d46014cc |
| moracheck_s1 | bc6086c9d78c0e846be7832aaa2d24f753c6df17b3fada44ec5053f69adb623e |
| moracheck_s2 | 07d8f5cd7b5537450b71a0cad429ed50486f273f0fce2a4940018999213a3c84 |

No learner speech or rating was used for training, model selection or the design of the simulators.

## Items and labels

- Main set: the difficult-sound sentences (B set). Each rated sentence contributes its target word; the canonical phonemes come from OpenJTalk, and the word score is the largest mora probability over the morae of the target word.
- Label: majority of the teachers who rated the item. Items with a 2 to 2 tie among four teachers are left out of the main analysis. A sensitivity analysis counts ties as mispronounced.
- Secondary set: the difficult-sound words (D set), 1 to 5 ratings, Spearman correlation between the mean rating and the word score.
- Native control: the same sentences read by the native speakers of the corpus; we report the share of native target words above each method's threshold.

## Split

Speakers are split into development (30%) and test (70%), stratified by first language, with seed 2026 (`scripts/umejrf_analyze.py`). The development speakers set the decision threshold of each method (Youden index) and the error rate of each group. Test speakers are used once.

## Comparisons

1. Main: MoraCheck (mean of the three seeds' word scores) against the likelihood ratio, GOP and free recognition, word AUC on test speakers.
2. Intervals: speaker-cluster bootstrap with 2,000 resamples for each AUC and for the paired difference to MoraCheck.
3. Agreement: Cohen's kappa at the development threshold. One teacher is held out at a time; that teacher and each method are compared with the majority of the remaining teachers, and the kappas are averaged.
4. Groups and first languages: AUC per error group and per first language, reported only where at least 30 positive and 30 negative test items exist.
5. Secondary model: MoraCheck retrained with the group mixture set to the development error rates (learner-informed), three seeds, compared with the frozen models on test speakers.
