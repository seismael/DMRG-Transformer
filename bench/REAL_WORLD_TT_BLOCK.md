# Softmax-Attention TTBlock — Real-Task Validation

**Device:** `device=cuda:0 (NVIDIA GeForce MX150, sm_61, 2.0 GiB)`

## Final test-set accuracy

| Model | Train acc | **Test acc** | Params | Wall (s) | Peak GPU (MiB) |
| :---- | --------: | -----------: | -----: | -------: | -------------: |
| TT-DMRG (no grads) | 0.8622 | **0.8611** | 1,946 | 43.82 | 370.9 |
| Dense (AdamW, MSE) | 0.9896 | **0.9806** | 1,946 | 49.43 | 301.6 |
| Dense (AdamW, CE)  | 1.0000 | **0.9667** | 1,946 | 49.29 | 301.5 |
| Large Dense (CE)   | 1.0000 | **0.9611** | 4,066 | 68.33 | 311.9 |

## Iso-time fairness check

| Comparison | Wall budget (s) | Test acc at budget | Final test acc |
| :--------- | --------------: | -----------------: | -------------: |
| TT-DMRG (reference) | 43.82 | **0.8611** | 0.8611 |
| Dense Adam-MSE      | 43.82 | **0.9806** | 0.9806 |
| Dense Adam-CE       | 43.82 | **0.9667** | 0.9667 |
| Large Dense-CE      | 43.82 | **0.9583** | 0.9611 |