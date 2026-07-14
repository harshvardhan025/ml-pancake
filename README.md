# Your First 8 Weeks — Day-by-Day Schedule
*Lane A (Application Engineer) · 10 hrs/week · covers Phase 0, Phase 1, and the start of Phase 2*

## How to use this
- **5 sessions/week.** Suggested rhythm: **Mon / Tue / Thu ≈ 1.5h each**, **Sat ≈ 3h**, **Sun ≈ 2.5h**. Remap the days to your life; keep the order.
- Every session has a **resource** (what to watch/read) and an **exercise** (what to produce). Don't just watch — type the code.
- Each week ends with a **deliverable**. If it doesn't work yet, spend buffer time before moving on.
- Keep one GitHub repo (`genai-journey`) and commit each week's notebook.

---

## WEEK 1 — Linear algebra intuition + NumPy *(Phase 0)*

**S1 · Mon** — 3Blue1Brown *Essence of Linear Algebra*, Ch.1 (Vectors) + Ch.2 (Linear combinations, span, basis).
*Exercise:* create vectors as NumPy arrays; add and scale them. Note in comments what a vector means in ML (a data example / an embedding).

**S2 · Tue** — 3B1B Ch.3 (Linear transformations) + Ch.4 (Matrix multiplication as composition).
*Exercise:* multiply two 2×2 matrices by hand, then verify with `@` in NumPy.

**S3 · Thu** — 3B1B Ch.9 (Dot products) — *this is the core of attention, take it seriously.*
*Exercise:* compute dot product and cosine similarity between vectors; try one similar pair and one dissimilar pair, observe the numbers.

**S4 · Sat** — NumPy official *"the absolute basics for beginners"* + *"quickstart"*: array creation, indexing, slicing, broadcasting, reshaping, aggregations.
*Exercise:* normalize each column of a matrix using vectorized ops only (no Python loops).

**S5 · Sun** — 3B1B Ch.14 (Eigenvectors/eigenvalues) for intuition only (don't stress mastery). Then consolidate.
*Deliverable:* a `linear-algebra-in-numpy.ipynb` notebook demonstrating vectors, matrix multiply, dot product, and cosine similarity, each with a one-line explanation.

---

## WEEK 2 — Calculus intuition + Pandas/Matplotlib *(Phase 0)*

**S1 · Mon** — 3B1B *Essence of Calculus* Ch.1 (the paradox of the derivative) + Ch.2 (derivatives geometrically).
*Note:* a derivative is "sensitivity / slope" — this is exactly what training uses to improve a model.

**S2 · Tue** — 3B1B Ch.3 (chain rule & product rule). **The chain rule is what backpropagation is** — rewatch until it's solid.
*Exercise:* hand-compute the derivative of a simple composite like `f(g(x))`.

**S3 · Thu** — 3B1B Ch.4 (derivatives of eˣ) + read a short piece on partial derivatives/gradients (Khan Academy "gradient").
*Note:* the gradient points in the direction of steepest increase — training walks the opposite way.

**S4 · Sat** — Pandas *"10 minutes to pandas."* Load a real CSV (Kaggle Titanic works well).
*Exercise:* filter rows, `groupby`, handle missing values, and call `.describe()`.

**S5 · Sun** — Matplotlib pyplot basics.
*Deliverable:* a `data-eda.ipynb` that cleans the CSV and plots histograms, a scatter, and a bar chart.

---

## WEEK 3 — Probability/stats intuition + consolidate Phase 0

**S1 · Mon** — Khan Academy: random variables, probability distributions (discrete vs continuous), expectation (mean), variance/std.

**S2 · Tue** — Khan Academy: conditional probability + Bayes' theorem.
*Note:* where Bayes shows up — classification and evaluation.

**S3 · Thu** — StatQuest: *"Entropy for Data Science"* / *"Cross Entropy"* + *"Bias and Variance."*
*Note:* cross-entropy is the loss you'll use for classification constantly.

**S4 · Sat** — Extend your Week-2 notebook: summary stats, a correlation heatmap, and short markdown notes interpreting the data.

**S5 · Sun** — Review + buffer. Rewatch anything shaky (usually the chain rule or dot products).
*Deliverable (ends Phase 0):* the polished EDA notebook **plus** a one-page "math intuition cheat sheet" *in your own words* covering: vector, matrix multiply, dot product, derivative, chain rule, gradient, distribution, cross-entropy.

---

## WEEK 4 — Machine Learning: regression *(Phase 1)*

Enroll: Andrew Ng **Machine Learning Specialization → Course 1: Supervised ML: Regression and Classification.**

**S1 · Mon** — Course 1, Week 1: supervised vs unsupervised, ML workflow, cost function intuition.

**S2 · Tue** — Course 1, Week 1 cont.: gradient descent. **Connect it to last week's calculus** — this is the chain rule + gradient in action. Do the labs.

**S3 · Thu** — Course 1, Week 2: multiple linear regression, vectorization (ties to your NumPy), feature scaling.

**S4 · Sat** — Course 1, Week 2 labs + quizzes. Then implement a small linear regression in scikit-learn on a toy dataset with a train/test split.

**S5 · Sun** — StatQuest: *"Linear Regression,"* *"Least Squares,"* *"R-squared."*
*Deliverable:* a scikit-learn linear-regression notebook predicting a numeric target, with a train/test split.

---

## WEEK 5 — Machine Learning: classification + evaluation *(Phase 1)*

**S1 · Mon** — Course 1, Week 3: logistic regression, decision boundary.

**S2 · Tue** — Course 1, Week 3 cont.: overfitting and regularization. Do the labs.

**S3 · Thu** — StatQuest: *"Logistic Regression,"* *"ROC and AUC,"* *"The Confusion Matrix,"* *"Cross Validation."*

**S4 · Sat** — scikit-learn: train a logistic-regression classifier (Titanic or similar); evaluate with accuracy, precision, recall, F1, and ROC-AUC.

**S5 · Sun** — StatQuest: precision vs recall — decide which matters for your problem and why.
*Deliverable:* a classifier with a full evaluation report + one paragraph justifying your metric choice.

---

## WEEK 6 — Machine Learning: the classic algorithms + bake-off *(Phase 1)*

**S1 · Mon** — StatQuest: *"Decision Trees"* + *"Random Forests Part 1."*

**S2 · Tue** — StatQuest: *"Support Vector Machines (Main Ideas)"* + *"K-nearest neighbors."*

**S3 · Thu** — StatQuest: *"K-means clustering"* (your first unsupervised method).

**S4 · Sat** — scikit-learn: compare logistic regression, random forest, SVM, and kNN on one dataset using cross-validation; tune one hyperparameter.

**S5 · Sun** — Write a short model-comparison summary: which won, on which metric, and why.
*Deliverable (ends Phase 1):* a "model bake-off" notebook with cross-validated metrics across several algorithms + a written conclusion.

---

## WEEK 7 — Deep Learning: intuition + PyTorch basics *(Phase 2)*

**S1 · Mon** — 3B1B *Neural Networks* video 1 (what is a neural network) + video 2 (gradient descent / how NNs learn).

**S2 · Tue** — 3B1B *Neural Networks* video 3 (backpropagation) + video 4 (backprop calculus). **Your chain-rule work from Week 2 now pays off.**

**S3 · Thu** — PyTorch official *"60 Minute Blitz"* — tensors + autograd sections.
*Exercise:* create tensors, run ops, call `.backward()`, inspect `.grad`.

**S4 · Sat** — PyTorch tutorials: `nn.Module`, loss functions, optimizers. Define a 2-layer MLP (untrained).

**S5 · Sun** — Load Fashion-MNIST via `torchvision`; build a `DataLoader`; view a few images with Matplotlib.
*Deliverable:* a PyTorch notebook with an autograd demo, a defined MLP, and a working DataLoader.

---

## WEEK 8 — Deep Learning: train your MLP *(Phase 2 — milestone)*

**S1 · Mon** — Write the training loop **manually**: forward pass → cross-entropy loss → `loss.backward()` → `optimizer.step()` → `optimizer.zero_grad()`. Understand every line.

**S2 · Tue** — Train on Fashion-MNIST; track loss per epoch; plot the loss curve.

**S3 · Thu** — Evaluate on the test set; compute accuracy; inspect a few misclassified images.

**S4 · Sat** — Experiment: change learning rate, hidden size, add a layer, add dropout. Watch for overfitting and note what each change does.

**S5 · Sun** — Clean up; write a README explaining your architecture, training loop, and results.
**Milestone deliverable:** *a from-scratch MLP in PyTorch with a manual training loop classifying Fashion-MNIST, including a loss curve and a written explanation of each hyperparameter's effect.*

---

## After Week 8
You'll be ~2 weeks into Phase 2. Continue: finish Phase 2 (weeks 9–10), then Phases 3+4 (NLP → Transformers → **your mini-GPT**) using Andrej Karpathy's *Neural Networks: Zero to Hero*. Ping me when you reach Week 8's deliverable and I'll build the next 8-week block (Phase 2 finish → Transformers → mini-GPT) the same way.

## If you fall behind
Don't skip — extend. It's better to take 10 weeks over this block and truly build the MLP than to rush to RAG on a shaky foundation. The weekend sessions have natural buffer; use them to catch up rather than pushing into new material.
