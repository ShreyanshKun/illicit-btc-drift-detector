# Viva preparation: questions and answers

Short, honest answers you can say in your own words. Numbers come from this project's actual results (test period = time steps 35–49 unless stated). Everyone in the team should be able to answer every question.

## A. The problem and the data

**1. In one minute, what is your project?**
We use the Elliptic Bitcoin dataset (about 204,000 transactions) to classify each transaction as illicit or licit. Because illicit cases are rare and criminals change behaviour, we avoid accuracy, split the data by time, compare a baseline with Logistic Regression, Random Forest and XGBoost using illicit-class F1 and PR-AUC, and study how performance collapses after time step 43 (concept drift) and whether retraining helps.

**2. Why this dataset?**
It is real (published by the blockchain-analytics company Elliptic), large enough for proper train/validation/test splits, comes with a peer-reviewed paper (Weber et al., 2019), and is a genuine security/finance problem with known benchmark results.

**3. What is a node and an edge here?**
A node is one Bitcoin transaction; an edge means coins flowed from one transaction into another. Illicit money often passes through many hops, so a transaction's neighbours say something about it.

**4. What are the 166 features?**
Time step + 93 *local* features (the transaction itself) + 72 *aggregated* features (summaries of its neighbours). They are anonymised, so **we cannot and do not say what any single column means**.

**5. 77% of the data is unlabelled. What did you do?**
We kept those rows out of training, validation, testing and every supervised metric. Treating "unknown" as licit would add label noise and make results look better than they are. We still used them for label-free work: graph degrees and measuring how inputs drift.

**6. Why is accuracy the wrong metric?**
Only 9.8% of labelled transactions are illicit (6.5% in the test period). Always answering "licit" gives 93.5% accuracy on our test period and catches **zero** crime. That is exactly what our majority baseline does.

## B. Cleaning and EDA

**7. Did you find missing values or duplicates?**
No missing values and no duplicate ids or edges. There are 34 duplicate *feature rows* (15 groups), all inside the same time step and none with conflicting labels, so we kept them; they cannot leak across the chronological split.

**8. Why did you not remove outliers?**
About 13% of a typical feature's values are beyond 3×IQR, but the boxplots show the extreme values are mostly *licit* transactions. They are real behaviour that helps separate classes, so deleting them would remove information. Trees ignore scale; for Logistic Regression we standardise and clip at ±5 SD.

**9. What was the most important EDA finding?**
The illicit share is not constant: it goes from 0.3% to 36% across steps, is 20% in validation and drops to 2.5% after step 43. The data is changing over time, which motivates the whole drift analysis.

**10. Did EDA show drift before you modelled?**
Yes. The Kolmogorov–Smirnov shift of the features against the training period rises from 0.113 (steps 1–27) to 0.166 (steps 43–49), and the share of features with KS above 0.2 doubles from 11% to 21%. That input drift is gradual (no jump at 43).

## C. Splitting and leakage

**11. How did you split the data and why?**
Chronologically: train steps 1–27, validation 28–34, test 35–49. A deployed model must score transactions after the ones it was trained on. We tune on validation and look at the test period once.

**12. What is data leakage and how did you avoid it?**
Leakage is when information that would not exist at prediction time reaches training. We prevent it by: removing unknown labels first, splitting strictly by time (asserted in code), fitting imputation/scaling inside pipelines on training rows only (unit-tested), tuning with expanding-window validation that never touches steps 35–49, choosing the threshold on validation, not using the time step as a feature, and never using neighbour *labels*. Edges only join transactions in the same step, so graph features do not use the future either.

**13. What did the random-split experiment show?**
With the same models, a random split gives F1 0.945 (Random Forest) and 0.955 (XGBoost) versus 0.757 and 0.776 on the temporal split; Logistic Regression goes from 0.766 to 0.255. A random split puts near-copies of test transactions in training, so it overstates performance by about 0.18 F1 for trees.

**14. What is expanding-window cross-validation?**
Folds respect time: train on steps 1–13 and validate on 14–20; train on 1–20 and validate on 21–27; train on 1–27 and validate on 28–34. Each fold trains on the past and validates on the future, unlike ordinary k-fold, which shuffles time.

## D. Models

**14b. Why these models?**
A majority baseline (floor), Logistic Regression (simple, interpretable linear model), Random Forest (non-linear, robust to heavy tails) and XGBoost (strong boosted trees with imbalance control). Each has a purpose; we did not add models just to look advanced.

**15. How do Random Forest and XGBoost work?**
A Random Forest trains many decision trees on random subsets and averages their votes, which reduces overfitting. XGBoost builds trees one after another, each correcting the errors of the previous ones. Both can capture non-linear interactions that a linear model cannot.

**16. How did you handle class imbalance?**
Class weights in Logistic Regression and Random Forest, `scale_pos_weight` (licit/illicit ratio from training labels) in XGBoost, illicit-class metrics, and a threshold tuned on validation data.

**17. What did hyper-parameter tuning change?**
Little for the tree models: across the whole grid the cross-validated PR-AUC varies by only 0.013 (Random Forest) and 0.005 (XGBoost). Logistic Regression is worse in every fold, so the gap to the trees is not a tuning artefact.

**18. Why didn't you build a graph neural network?**
It was optional in the brief and we prioritised correctness and explainability of the core pipeline. We therefore **cannot** claim whether a GNN would beat the tree models. What we did test: local features alone give test PR-AUC 0.77/0.75 (RF/XGBoost); aggregated neighbour features alone give 0.61/0.63; both together 0.78/0.78. So the supplied graph information helps a little (+0.01 to +0.03) but local features carry most of the signal.

## E. Evaluation

**19. Explain precision, recall, F1 and PR-AUC.**
Precision: of what we flag as illicit, how much really is (few false alarms). Recall: of all illicit transactions, how many we caught. F1: their harmonic balance. PR-AUC: area under the precision–recall curve; threshold-free and the standard summary for imbalanced data. We report them for the illicit class only.

**20. Why is ROC-AUC only supplementary?**
With very many licit transactions the false-positive *rate* looks tiny even when thousands of false alarms swamp the true detections (Logistic Regression has ROC-AUC 0.86 but precision 0.15).

**21. What are your test results?**
Random Forest: precision 0.796, recall 0.722, F1 0.757, PR-AUC 0.784. XGBoost: 0.914, 0.675, 0.776, 0.783. Logistic Regression: 0.148, 0.920, 0.255, 0.257. Majority baseline: all zeros except 93.5% accuracy.

**22. Which model is best? Be honest.**
By the rule we fixed *before* looking at the test period (validation PR-AUC and F1) Random Forest is best. On the test period Random Forest and XGBoost tie on PR-AUC (bootstrap interval of the difference includes zero); XGBoost has higher F1 and precision with fewer false alarms (69 vs 201), Random Forest has higher recall (fewer misses: 301 vs 352). We use XGBoost as the demo default because of fewer false alarms and exact per-prediction explanations, and we state that this choice was made after seeing test results.

**23. False positives versus false negatives: which matters more?**
It depends on costs. A false negative is a missed illicit transaction; a false positive wastes an investigator's time and can wrongly flag an innocent user. Raising the threshold trades misses for fewer false alarms; our validation-tuned threshold maximises F1 as a compromise. The right operating point depends on investigator capacity.

**24. Was the test set used to choose anything?**
No model, hyper-parameter, feature set or threshold was chosen on test data. Afterwards we *report* additional analyses on the test period (ablations, drift, errors), and the walk-forward retraining experiment legitimately trains on earlier test steps, because that is how deployment would work.

## F. Concept drift

**25. What is concept drift and what did you find?**
Concept drift means the relationship between inputs and the label changes over time, so a model trained on the past stops working. Our models work well in steps 35–42 (per-step F1 0.63–0.95) but collapse at step 43: the static XGBoost model catches only 3 of the 169 illicit transactions in steps 43–49, versus 728 of 914 before.

**26. Is the darknet-market shutdown the cause?**
It is a *plausible hypothesis*, not something our data proves. We observe **when** behaviour changes, not **why**.

**27. Why do you think it is concept drift rather than just input drift?**
Input drift (KS statistic) is gradual and has no jump at 43, yet performance collapses abruptly, so what changed seems to be how features relate to the label. The missed illicit transactions got very low illicit probabilities (79% below 0.1): the model is confidently wrong, not unsure. This is an inference, not a proof.

**28. Does retraining solve it?**
It helps but does not solve it. Retraining on recent labelled data raises pooled XGBoost F1 from 0.771 to 0.873 and recall after step 43 from 0.018 to 0.391, but F1 after 43 is still only about 0.54 (versus about 0.9 before). It needs a steady supply of fresh confirmed labels, which is optimistic (we assumed the previous step's labels are available). Sliding and expanding windows perform alike.

**29. Did removing unstable features help?**
A little: dropping the 10% most-shifted features (chosen without test data) improved validation and test F1 by about 0.015–0.03; removing 50% hurt XGBoost. It does not fix the collapse after step 43.

**30. Why is Random Forest worse than XGBoost after retraining at threshold 0.5?**
Its PR-AUC rises (0.045 → 0.565 after step 43) but its probabilities stay low, so a fixed 0.5 threshold still misses almost everything. Its *ranking* adapts; its calibration/threshold does not. A drift-aware threshold would likely help.

## G. Features and errors

**31. Which features matter?**
XGBoost leans on `local_41` (18% of gain), `local_53`, `local_47`, `agg_10`, `agg_7`. Methods disagree about which member of a correlated group matters, so we treat rankings as indicative. We **do not know what these columns mean** and never invent meanings.

**32. Do important features stay important?**
No. After step 43 the permutation importance of `local_59` halves, `local_52` and `agg_70` fall to about zero and `local_41` turns negative (shuffling it slightly improves PR-AUC). Importance rankings of models trained on different periods correlate only 0.32–0.40.

**33. What do the errors look like?**
Of XGBoost's 352 misses, 166 (47%) are in steps 43–49, which hold only 16% of the illicit test cases. False alarms are rare and stable (69 total). Random Forest and XGBoost miss the same transactions (all of RF's 301 misses are also XGBoost misses), which points to a data-level cause rather than a model quirk.

**34. Did engineered features help?**
No consistently. Cross-validated PR-AUC was 0.9464 (original) vs 0.9455 (with degrees and neighbour means), so we kept the original 165 features. Test PR-AUC was essentially unchanged either way (0.784 vs 0.786 for RF).

## H. Reproducibility, application and ethics

**35. Can someone reproduce your numbers?**
Yes. Seeds are fixed (42), requirements are pinned, `python train.py` then `python run_analysis.py` regenerate everything, and we verified that re-fitting the saved configurations reproduces the stored predictions exactly (maximum difference 0.0). 17 unit tests cover loading, splitting, scaling, feature engineering, metrics and prediction validation.

**36. How does the app work?**
It loads saved models and result tables; nothing is retrained at start-up. Users can upload a CSV with the 165 feature columns (validated; batch scoring; download), pick a sample transaction or edit features, and see the risk score, predicted class and (for XGBoost) per-feature contributions. Every prediction carries a disclaimer.

**37. What are the limitations?**
Anonymised features; 77% unlabelled and possibly biased labels from one company; a single time window; the step-43 cause is a hypothesis; late steps contain few illicit cases so per-step scores are noisy; the validation period has 20% illicit vs 6.5% in test (threshold transfer); retraining assumes immediate labels; GNN/MLP/Elliptic++ were not done.

**38. Ethical considerations?**
A model like this can wrongly flag innocent users. A score is a statistical signal, not proof, and should support human investigators rather than replace them.

**39. What would you do next?**
Train GCN/GraphSAGE for a direct tree-vs-graph comparison, add the Elliptic++ wallet-address data, build automatic drift detection that triggers retraining, calibrate probabilities and adapt the threshold to the changing base rate, and use the unlabelled data semi-supervised.

**40. What if the examiner says "your result contradicts expectations"?**
We stated hypotheses as expectations to test. Where results differ (engineered features did not help; Random Forest at 0.5 did not recover after retraining; the best model depends on the criterion), we report that honestly; that is still a valid finding.
