# Module 4 — Evaluation, Bayesian Inference, Information Theory, Explainability

The single-line mapping to memorize:

> "Module 4 shows up in `src/evaluation/statistical.py` and `src/evaluation/explainability.py`. Frequentist tests, Bayesian update, mutual information, SHAP for attack prediction, Shapley values for defense attribution — all five M4 concepts (eval, calibration, Bayesian, IT, XAI) are concretely there."

This is the densest section. The panel will probe stats deeply — derive everything you put on paper.

---

## 1. Statistical tests — when to use which

| Question | Test | Code |
|----------|------|------|
| Is ASR different between model A and model B? (independent samples) | Chi-squared on 2×2 contingency | `chi_squared_test()` :13 |
| Is detection rate different between defense A and defense B on the *same* attacks? (paired) | McNemar's test | `mcnemar_test()` :42 |
| What's the uncertainty around a single ASR estimate? | Wilson score CI | `confidence_interval()` :73 |
| How large is the effect, not just is it significant? | Cohen's h | `effect_size_cohens_h()` :104 |
| Comparing many groups — control FWER | Bonferroni correction | `bonferroni_correction()` :119 |
| Bayesian estimate of vulnerability with credible interval | Beta-Binomial | `bayesian_vulnerability()` :233 |
| How predictive is attack category of success? | Mutual information | `mutual_information()` :323 |
| Is the model uniformly vulnerable or focused on a few categories? | Entropy of success distribution | `entropy_of_vulnerability_profile()` :395 |
| Fair attribution of detection power to each defense in a stack | Shapley values | `defense_shapley_values()` :455 |

---

## 2. Chi-squared test (2×2 contingency)

### When
Compare success rates between two *independent* groups. Example: ASR of Llama-3.3 vs ASR of Qwen-3 across the same attack set.

### Derivation
Observed contingency table:
```
            success   fail
model_a       a         b
model_b       c         d
```
Under H0 (rates equal), expected count in cell (i,j) is:
$$E_{ij} = \frac{R_i \cdot C_j}{N}$$
where $R_i, C_j$ are row/column totals and $N$ is grand total.

Statistic:
$$\chi^2 = \sum_{i,j} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}$$

Distributed as $\chi^2_{(r-1)(c-1)} = \chi^2_1$ for a 2×2 table. The code uses `scipy.stats.chi2_contingency` which applies Yates' continuity correction by default for 2×2.

### Assumption
Independent samples. Each (attack, model) result is one observation. Breaks if you've reused the same attack against multiple defenses with the same model — that's *paired*, not independent. Use McNemar instead.

---

## 3. McNemar's test (paired)

### When
Two defense configurations evaluated on the *same* attacks. Example: input_filter alone vs input_filter + semantic_filter — do they differ in detection rate?

### Derivation
Paired contingency:
```
                            defense_b detects   defense_b misses
defense_a detects                  a                  b
defense_a misses                   c                  d
```
- $a, d$ are *concordant pairs* (both agree) — uninformative.
- $b, c$ are *discordant pairs* — these are what we test.

Under H0 (defenses equally likely to detect any given attack), discordant pairs are split equally: $b \sim \text{Binomial}(b+c, 0.5)$.

McNemar statistic (with continuity correction):
$$\chi^2 = \frac{(|b - c| - 1)^2}{b + c}$$
Distributed as $\chi^2_1$. The code:
```python
b = ((df.detected_a == True) & (df.detected_b == False)).sum()
c = ((df.detected_a == False) & (df.detected_b == True)).sum()
statistic = (abs(b - c) - 1) ** 2 / (b + c)
p_value = stats.chi2.sf(statistic, df=1)
```

### Why paired is more powerful
Chi-squared on the same data would lose power because it treats each attack as independent, ignoring the natural pairing. McNemar conditions on the matched structure, which removes between-attack variance from the noise floor.

---

## 4. Wilson score confidence interval

### When
You have $k$ successes in $n$ trials and want a CI on $p$. The standard "Wald" CI ($p \pm z\sqrt{p(1-p)/n}$) is bad for small $n$ or extreme $p$ (can give intervals outside [0,1] and is anti-conservative).

### Derivation (sketch)
Wilson inverts the score test (not the Wald test) — i.e., he solves for the $p$ values that would *not* reject H0 at level $\alpha$:
$$\left| \frac{\hat{p} - p}{\sqrt{p(1-p)/n}} \right| \le z_{\alpha/2}$$

Squaring and solving the quadratic in $p$ gives the Wilson endpoints:
$$\text{CI} = \frac{\hat{p} + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + z^2/n}$$

The code computes this directly (`center` and `spread`).

### Why this matters for the viva
Many of our ASRs are small ($k = 1, 2$ out of $n = 4$ to 12). The Wilson CI is the right tool: it's a coverage-correct interval that doesn't collapse to width 0 when $k = 0$ or $k = n$.

### Concrete example from our results
v8.2 GCG: 1 success out of 4 attacks on groq-qwen → $\hat{p} = 0.25$, Wilson 95% CI = (0.046, 0.700). This wide CI is exactly why we say "the effect size, not p-value, is the load-bearing claim" in §5.3 of the findings report — with $n=4$ you cannot rule out true rates between 5% and 70%.

---

## 5. Cohen's h — effect size for proportions

### Derivation
Arcsine-square-root transform:
$$\varphi = 2\arcsin\sqrt{p}$$
This is the *variance-stabilizing* transform for the binomial — $\text{Var}(\varphi) \approx 1/n$ regardless of $p$. So differences in $\varphi$ are comparable across the [0,1] range.

$$h = \varphi_a - \varphi_b = 2\arcsin\sqrt{p_a} - 2\arcsin\sqrt{p_b}$$

### Interpretation
$|h| < 0.2$ small, 0.2–0.5 medium, 0.5–0.8 medium-large, $|h| > 0.8$ large.

### Why this matters
P-values depend on $n$; you can always get $p < 0.05$ with enough samples even for a trivial effect. Cohen's h is sample-size-independent. When you compare 75% detection (in-domain) vs 25% detection (AdvBench), Cohen's h is:
$$h = 2\arcsin\sqrt{0.75} - 2\arcsin\sqrt{0.25} = 2(1.047) - 2(0.524) = 1.05$$
Large effect by Cohen's convention. The viva framing: "even with small samples, the effect size between in-domain and out-of-domain training is large — h > 1."

---

## 6. Bonferroni correction

### When
Multiple pairwise comparisons. Family-wise error rate (FWER): probability that *any* of $k$ tests gives a false positive under H0 grows as $1 - (1-\alpha)^k$.

### Bonferroni
$$\alpha_{\text{adj}} = \alpha / k, \quad p_{\text{adj}} = \min(k \cdot p, 1)$$

The code does the latter formulation — adjusts the p-values rather than the threshold.

### Why this matters
We compare 4 models × 7+ defenses → 28+ pairwise comparisons against a baseline. Without correction, ~1.4 false positives expected at α=0.05. With Bonferroni, controlled at α=0.05 across the family. Bonferroni is conservative — alternative is Benjamini-Hochberg (controls FDR not FWER), which is more powerful but harder to defend in a high-stakes domain like financial-system security where false negatives are bad.

---

## 7. Bayesian vulnerability — Beta-Binomial

### The model (likely to be asked verbatim)
Prior over vulnerability rate $p$:
$$p \sim \text{Beta}(\alpha, \beta)$$
Likelihood given $k$ successes in $n$ attacks:
$$k \mid n, p \sim \text{Binomial}(n, p)$$
Posterior (conjugate update — this is why we picked Beta):
$$p \mid k, n \sim \text{Beta}(\alpha + k, \beta + n - k)$$

### Derivation of conjugacy (be ready to do this)
$$P(p \mid k, n) \propto P(k \mid n, p) \cdot P(p)$$
$$\propto \binom{n}{k} p^k (1-p)^{n-k} \cdot \frac{p^{\alpha-1}(1-p)^{\beta-1}}{B(\alpha,\beta)}$$
$$\propto p^{\alpha+k-1}(1-p)^{\beta+n-k-1}$$
which is the kernel of $\text{Beta}(\alpha+k, \beta+n-k)$.

### Posterior summaries
- **Mean:** $\frac{\alpha + k}{\alpha + \beta + n}$
- **Mode (MAP):** $\frac{\alpha + k - 1}{\alpha + \beta + n - 2}$ (when both shape params > 1)
- **Variance:** $\frac{(\alpha+k)(\beta+n-k)}{(\alpha+\beta+n)^2(\alpha+\beta+n+1)}$
- **95% credible interval:** $[F^{-1}(0.025), F^{-1}(0.975)]$ where $F$ is the Beta CDF.

### Why Bayesian here?
Two reasons. First, **small-sample regularization** — with $n = 4$ and $k = 0$, the frequentist MLE is $\hat{p} = 0$ which is a terrible estimate. The Beta(1,1) prior shrinks this to a posterior mean of $1/6$, which is more honest about our uncertainty. Second, **direct probability statements** — frequentist CIs require contorted "if we repeated the experiment infinitely..." language. The posterior lets us say `P(vulnerability > 30%) = 0.42` directly, which the function returns as `prob_vulnerability_gt_30pct`.

### Bayes factor (Savage-Dickey)
At the point $p_0 = 0.3$, comparing H1 ($p > 0.3$) vs H0 ($p = 0.3$):
$$\text{BF}_{10} = \frac{P(p_0 \mid H_0, \text{prior})}{P(p_0 \mid \text{data})} = \frac{\text{prior density at } p_0}{\text{posterior density at } p_0}$$

> 1 = data shifts mass toward H1. The code computes this as `bayes_factor_gt_30pct`.

---

## 8. Information theory — mutual information and entropy

### Mutual information
$$I(X; Y) = \sum_{x,y} P(x,y) \log_2 \frac{P(x,y)}{P(x) P(y)}$$

Equivalent forms:
- $I(X;Y) = H(Y) - H(Y|X) = H(X) - H(X|Y)$ (information gain framing)
- $I(X;Y) = D_{KL}(P(X,Y) \| P(X)P(Y))$ (KL from joint to product of marginals)

**Normalized MI:**
$$\text{NMI} = \frac{I(X;Y)}{\min(H(X), H(Y))} \in [0, 1]$$

### What we use it for
Compute $I(\text{category}; \text{success})$: how much does knowing the attack category tell us about success? If $I \approx 0$, all categories are equally (un)successful — no information. If $I$ is large, some categories are strongly predictive.

### Entropy of vulnerability profile
$$H(\text{successes per category}) = -\sum_c P(c) \log_2 P(c)$$

Where $P(c)$ is the share of successful attacks coming from category $c$. Low entropy = model has a concentrated weak point (one category dominates). High entropy = uniformly vulnerable across categories.

**The viva framing:** "A model with low vulnerability entropy is *easier to defend* because you only need to fix the concentrated category. A model with high vulnerability entropy is *harder to defend* because there's no single weak point."

---

## 9. SHAP — explaining attack success predictions

**Code:** `src/evaluation/explainability.py:83`

### What it does
Trains an XGBoost classifier on attack metadata (category, severity, model, defense, has_tool_overrides, has_context_injection) to predict `success`. Then uses SHAP to attribute each prediction to each feature.

### SHAP value (from cooperative game theory)
For prediction $f(x)$ and feature $i$:
$$\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|! (|N| - |S| - 1)!}{|N|!} \left[ f(S \cup \{i\}) - f(S) \right]$$

This is the marginal contribution of feature $i$ averaged over all possible orderings in which it could enter the coalition.

### Why SHAP is uniquely justified (vs gain importances)
SHAP values are the *unique* feature attribution satisfying four axioms (Lundberg & Lee 2017):
1. **Efficiency** — attributions sum to $f(x) - E[f(x)]$.
2. **Symmetry** — equally-contributing features get equal attribution.
3. **Null** — features that never change the prediction get zero.
4. **Additivity** — consistent across ensemble models.

XGBoost's built-in `gain` importance lacks these guarantees (it's frequency-of-use weighted by gain, not a fair attribution).

### Computational note
Exact Shapley is $O(2^n)$. For tree ensembles, TreeSHAP (Lundberg 2018) computes exact Shapley in $O(TLD^2)$ where $T$=trees, $L$=leaves, $D$=depth. That's what the `shap` library uses.

---

## 10. Shapley values for defense attribution (different game)

**Code:** `src/evaluation/statistical.py:455`

Same Shapley formula, different game:
- **Players:** defenses.
- **Coalition:** a subset $S$ of defenses.
- **Value function $v(S)$:** detection rate (or $1 - \text{ASR}$) achievable with that coalition.
- **Shapley value for defense $i$:** its fair contribution to overall detection.

With 5 defenses, $5! = 120$ permutations — fully tractable, exact computation. The code iterates over all permutations and accumulates marginal contributions.

### Property
Sum of Shapley values = $v(\text{all defenses}) - v(\emptyset)$ = (detection rate with all defenses) - (detection rate with none) = total detection capability gained from the defense stack.

### Why this matters for the viva
Most ensemble systems report "the ensemble gets 80% detection." That doesn't tell you which defense is doing the work. Shapley attribution lets us say "input_filter contributes 0.25 of that 80%, semantic_filter 0.20, perplexity_filter 0.15, ..." — fair, axiomatic, decision-useful for "should we drop the most expensive defense?"

---

## Flashcards

**Q1.** Why McNemar instead of chi-squared when comparing two defenses?

> Because the comparison is paired — the *same* attacks are run against both defenses. Chi-squared on these data treats the (defense, attack) results as independent, which loses power because between-attack variance is part of the residual. McNemar conditions on the matched structure (only discordant pairs contribute to the statistic), removing that nuisance variance.

**Q2.** Derive the Beta-Binomial posterior.

> Bayes: $P(p|k,n) \propto P(k|p,n) P(p) = \binom{n}{k}p^k(1-p)^{n-k} \cdot p^{\alpha-1}(1-p)^{\beta-1}/B(\alpha,\beta) \propto p^{\alpha+k-1}(1-p)^{\beta+n-k-1}$, which is $\text{Beta}(\alpha+k, \beta+n-k)$ up to the normalizing constant.

**Q3.** What does "Wilson is variance-corrected" mean?

> The Wald CI uses $\hat{p}(1-\hat{p})/n$ as variance, plugging the *estimate* into the variance formula. This is anti-conservative when $\hat{p}$ is near 0 or 1, because the estimate is unstable. Wilson inverts the *score test*, which uses the variance under the *null* — gives intervals that respect [0,1], have correct nominal coverage for moderate $n$, and don't collapse to width 0 at $\hat{p}=0$ or 1.

**Q4.** A prof asks: "your $n=4$ result is meaningless." How do you respond?

> Two-part. (1) Yes, the *power* to detect cross-pair transfer at $n=4$ is low — Fisher's exact on a 0/4 vs 0/4 contingency is not significant. The 95% Wilson CI on a 0/4 success rate is (0, 0.60). I report this explicitly in §5.3 of the findings report and call it a primary limitation. (2) But the *effect size* claim (Cohen's h between in-domain and out-of-domain training detection is >1) is sample-size-independent. The headline isn't "transfer is significantly zero" — it's "the directional finding (cross-family transfer fails, intra-family succeeds at 50%) is consistent across two independent runs and matches the Zou et al. prior."

**Q5.** Why is Bonferroni "conservative" — and is that bad?

> Conservative means it controls FWER at $\alpha$ but is *not the tightest* control — i.e., it errs toward more false negatives in exchange for fewer false positives. Bonferroni assumes independent tests, which gives a worst-case bound. Holm-Bonferroni and Benjamini-Hochberg are tighter for correlated tests. We use Bonferroni because (a) it's the canonical baseline; (b) our tests are not strictly independent (same attacks across defenses), but Bonferroni still controls FWER under arbitrary dependence — so it's a safe choice; (c) in a security domain, a false positive (claiming a defense works when it doesn't) is worse than a false negative.

**Q6.** Difference between $H(Y)$, $H(Y|X)$, and $I(X;Y)$?

> $H(Y)$ = entropy of $Y$ = uncertainty about $Y$ alone. $H(Y|X)$ = conditional entropy = average uncertainty about $Y$ once $X$ is known. $I(X;Y) = H(Y) - H(Y|X)$ = the *reduction* in uncertainty about $Y$ from knowing $X$ — i.e., the information $X$ provides about $Y$.

**Q7.** Why are SHAP values "unique"?

> Shapley values are the unique feature attribution function satisfying efficiency (attributions sum to prediction minus baseline), symmetry (equal contributors get equal attribution), null (irrelevant features get zero), and additivity (consistent across ensembles). No other attribution method satisfies all four. The "uniqueness" theorem is from Shapley 1953 (cooperative game theory); Lundberg & Lee 2017 applied it to ML interpretability.

**Q8.** What's the difference between SHAP and gain-based feature importance?

> Gain importance counts how much each feature reduces loss when it splits a node, summed across the ensemble. This double-counts highly-correlated features and is biased toward high-cardinality features. SHAP is a *per-prediction* attribution averaged over all coalitions, which respects the four axioms. Practical difference: gain says "this feature was used often"; SHAP says "this feature *would change* the prediction by this much, on average, when it enters the model."

**Q9.** Walk through computing Bayes factor at $p_0 = 0.3$.

> The Savage-Dickey density ratio: $\text{BF}_{10} = \pi(p_0)/\pi(p_0 \mid \text{data})$ where $\pi$ is the prior density. With Beta(1,1) prior (uniform on [0,1]), $\pi(0.3) = 1$. After observing $k=2, n=4$ → posterior is Beta(3, 3), and $\pi(0.3 | \text{data}) = \text{Beta}(0.3; 3, 3) \approx 1.94$. So $\text{BF}_{10} = 1/1.94 \approx 0.52$ — slight evidence *against* H1 ($p > 0.3$) because the posterior is more peaked at 0.3 than the prior was. Interpretation: data didn't strongly shift us away from $p_0$.

**Q10.** Why use Shapley values for defense attribution when XGBoost already gives feature importances?

> Different game. XGBoost feature importances attribute the model's prediction to its *input features* — they answer "which input feature drove the prediction?" Shapley values for defenses answer a *different* question: "in the cooperative game of detection, what is each defense's fair contribution to the overall detection rate?" The "players" are defenses (not features) and the "value function" is detection rate of a coalition (not model logit). Same formula, different game.

**Q11.** A prof says "your effect sizes look big but your sample sizes are tiny — is this real or noise?" Best answer?

> "Honest answer: at $n = 4$ per pair, I cannot rule out that specific point estimates are noise. But the *directional pattern* is consistent across two independent runs (2026-05-14 AM gpt2-xl and PM Vicuna-7b) and matches the published Zou et al. 2023 Table 2 prior (~30-40% intra-family transfer). The directional claim — that family of training pipeline matters more than architecture — is *robust* even if specific ASR numbers have wide CIs. The right next step, which is in §8 future work, is increasing $n$ from 4 to 20+ attacks per pair."

**Q12.** Why didn't you use a permutation test instead of chi-squared?

> For 2×2 tables, the exact equivalent of chi-squared is Fisher's exact test (also conditions on margins). Permutation tests are more general but here would give essentially the same answer as Fisher's exact, at higher computational cost. For larger contingency tables or for ASR comparisons stratified by attack category, a permutation test would be the cleaner choice — it's in the future-work list.
