# Softmax-Attention TTBlock — Real-Task Validation

**Device:** `device=cuda:0 (NVIDIA GeForce MX150, sm_61, 2.0 GiB)`

## Final test-set accuracy

| Model | Train acc | **Test acc** | Params | Wall (s) | Peak GPU (MiB) |
| :---- | --------: | -----------: | -----: | -------: | -------------: |
| TT-DMRG (no grads) | 0.8629 | **0.8472** | 1,946 | 62.18 | 362.4 |
| Dense (AdamW, MSE) | 0.9896 | **0.9806** | 1,946 | 76.79 | 301.6 |
| Dense (AdamW, CE)  | 1.0000 | **0.9667** | 1,946 | 82.43 | 301.5 |
| Large Dense (CE)   | 1.0000 | **0.9611** | 4,066 | 99.04 | 311.9 |

## Iso-time fairness check

| Comparison | Wall budget (s) | Test acc at budget | Final test acc |
| :--------- | --------------: | -----------------: | -------------: |
| TT-DMRG (reference) | 62.18 | **0.8472** | 0.8472 |
| Dense Adam-MSE      | 62.18 | **0.9833** | 0.9806 |
| Dense Adam-CE       | 62.18 | **0.9667** | 0.9667 |
| Large Dense-CE      | 62.18 | **0.9583** | 0.9611 |