# Model And Hyperparameter Notes

## Main Model
`ARC_FX` is our model implemented as described in the paper. 

Headline setting:

- Currency universe: `USD + 10 non-USD currencies`
- Lookback window: `10`
- Graph sparsity: `top-k = 3`
- Hidden size: `48`
- Graph rank: `8`
- Dropout: `0.25`
- Edge dropout: `0.05`
- Optimizer: `AdamW`
- Learning rate: `3e-4`
- Weight decay: `1e-4`
- Batch size: `128`
- Max epochs: `80`
- Early stopping patience: `10`

## Shared Training Objective Used In The Reported Results

All reported predictive tables in this anonymous release use the same directional-core training objective as the common comparison anchor.

Specifically, the shared core objective is:

- active-set filtering with `A = {(b, i) : |y_{b,i}| >= tau}`
- threshold defined by `tau = Q0.40(|y_train_norm|)`
- directional softplus loss on the active set only

In words, all reported experiments are aligned around the same `Q0.40` active directional core, so very small normalized returns are excluded from the core directional loss and the model is trained to focus on economically more meaningful directional moves.

## Baselines

Baselines included in this anonymous package:

- `MLP`
- `Transformer`
- `GNN`
- `Corr-LSTM-GAT`
- `FXRP`

Shared experimental protocol:

- Same currency universe
- Same lookback window
- Same train/validation/test split
- Same optimizer family and early-stopping logic for trainable neural models
- Same seed set: `21, 42, 456`
- Same comparison loss: the `Q0.40` active directional core described above


## Baseline Implementations And Hyperparameters

Unless otherwise noted, the trainable baselines in the public comparison use the following shared training setup:

- Hidden size: `48`
- Dropout: `0.25`
- Optimizer: `AdamW`
- Learning rate: `3e-4`
- Weight decay: `1e-4`
- Batch size: `128`
- Max epochs: `80`
- Early stopping patience: `10`
- Selection metric: `mse_hit`
- Seed set: `21, 42, 456`

### Corr-LSTM-GAT

Implementation:
- Based on Landmesser-Rusek and Orłowski (2026)
- Static graph built from train-period cross-currency return correlations only
- Correlation threshold: `0.7`
- Two-layer LSTM temporal encoder followed by GAT message passing on the fixed graph
- Canonical ordering used in this package: `lstm_then_gat`

Implementation detail:

- The model is implemented as a hybrid temporal-relational baseline.
- For each non-USD currency, the full lookback sequence is first encoded by a two-layer LSTM.
- The resulting currency embeddings are then passed through graph attention layers on a fixed cross-currency graph.
- The fixed graph is not learned end-to-end during training. Instead, it is computed once from the training split only, using the correlation matrix of non-USD target returns.
- An undirected support is created by thresholding train-period correlations at `0.7`, and self-loops are then added to stabilize node updates.
- After graph attention, a scalar head produces one USD-relative score per non-USD currency, and a zero USD anchor is inserted before USD pinning.

What was adapted for this repository:

- The original idea of combining temporal recurrence with graph-based spillover modeling is preserved.
- The graph is constructed only from the training split so that no future information leaks into the relational structure.
- The implementation is adapted to the present task formulation, where the public benchmark predicts a vector of USD-relative returns rather than an arbitrary bilateral edge surface.
- The final readout therefore returns one score per currency relative to USD, which is the common output convention used by all models in this repository.
- The code also fixes the architectural order to `lstm_then_gat` for consistency across seeds and to avoid introducing an extra tuning dimension in the public comparison.

What was kept fixed:

- Recurrent temporal encoding
- Attention-based graph message passing
- Static correlation-driven support
- Shared hidden-size and optimizer protocol used in the common benchmark

Used hyperparameters:

- Effective hidden size: `48`
- Dropout: `0.25`
- Attention heads: `4`
- Graph source: training-period correlation graph
- Correlation threshold: `0.7`
- Architecture order: `lstm_then_gat`
- Lookback: `10`

### FXRP

Implementation:

- FXRP-style implementation based on Hong and Klabjan (2025)
- Edge-centric message-passing architecture
- Builds node summaries and bilateral edge summaries from rolling windows
- Alternating node-update and edge-update blocks
- Final prediction reads the directed edge from each currency to USD

Implementation detail:

- FXRP is implemented as an edge-centric graph neural baseline rather than a node-only encoder.
- Each currency first produces a node summary from the recent history window.
- In parallel, the model builds bilateral edge summaries from the USD-relative return history transformed into a bilateral return panel.
- The architecture then alternates between node updates and edge updates across multiple message-passing layers.
- Node updates aggregate transformed incoming edge information, while edge updates recombine source-node, edge-state, and destination-node information.
- The final prediction is produced by a linear edge head applied to the last edge state.
- In this repository, the reported USD-relative forecast for currency `i` is read directly from the learned directed edge `i -> USD`.

What was adapted for this repository:

- The original edge-centric spirit of FXRP is maintained: the model still reasons through evolving edge states rather than relying only on node embeddings.
- The public benchmark in this repository evaluates USD-relative next-period forecasts, so the output layer is adapted to read the canonical directed edge to USD.
- The implementation uses a fixed set of internal rolling windows `(1, 3, 5, 10)` to summarize short- and medium-horizon temporal information inside node and edge features.
- The final output head is kept activation-free so that predicted returns can be positive or negative without sign clipping.
- The model is trained under the same benchmark protocol as the other baselines, using the shared directional-core objective and the same data split, optimizer, and early-stopping procedure.

What was kept fixed:

- Edge-centric message-passing formulation
- Alternating node/edge update structure
- Multi-layer propagation depth
- Directed USD-relative readout within a common forecasting interface

Used hyperparameters:

- Effective hidden size: `48`
- Dropout: `0.25`
- Message-passing layers: `3`
- Window set inside the model: `(1, 3, 5, 10)`
- Lookback: `10`

## Reference Baselines

`Corr-LSTM-GAT` and `FXRP` should be read as reference baselines.

- `Corr-LSTM-GAT` is a static-correlation graph recurrent baseline.
- `FXRP` is an edge-centric message-passing baseline adapted to the present multi-currency USD-relative forecasting setup.
- In both cases, the main architectural idea is preserved, while the output interface and training wrapper are aligned to the common USD-relative forecasting task used in this repository.

They are included to broaden the architecture comparison beyond standard sequence models.


### MLP

Implementation:

- Shared cross-currency MLP over the flattened lookback window
- Each currency sequence is flattened from `[lookback, feature_dim]` into one vector
- Two hidden MLP blocks with `GELU` and dropout
- One scalar output per currency, then USD pinning

Used hyperparameters:

- Hidden size: `48`
- Dropout: `0.25`
- Lookback: `10`

### Transformer

Implementation:

- Time-wise transformer encoder applied independently to each currency sequence
- Input projection to hidden dimension
- Learned positional embedding over the lookback window
- Transformer encoder followed by temporal attention pooling
- One scalar output per currency, then USD pinning

Used hyperparameters:

- Hidden size: `48`
- Dropout: `0.25`
- Attention heads: `4`
- Transformer layers: `2`
- Lookback: `10`

### GNN

Implementation:

- GRU temporal encoder per currency
- Static learned sparse cross-currency adjacency
- Top-`k` neighbor selection from learned adjacency logits
- One round of message passing followed by prediction head

Used hyperparameters:

- Hidden size: `48`
- Dropout: `0.25`
- Graph top-`k`: `3`
- Lookback: `10`
