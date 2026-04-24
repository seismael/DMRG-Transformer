# Softmax-Attention TTBlock — Real-Task Validation

**Device:** `device=cuda:0 (NVIDIA GeForce MX150, sm_61, 2.0 GiB)`

## Final test-set accuracy

| Model | Train acc | **Test acc** | Params | Wall (s) | Peak GPU (MiB) |
| :---- | --------: | -----------: | -----: | -------: | -------------: |
| TT-DMRG (no grads) | 0.8546 | **0.8667** | 1,946 | 20.77 | 115.7 |
| Dense (AdamW, MSE) | 0.9896 | **0.9806** | 1,946 | 62.52 | 41.6 |
| Dense (AdamW, CE)  | 1.0000 | **0.9667** | 1,946 | 66.05 | 41.5 |
| Large Dense (CE)   | 1.0000 | **0.9611** | 4,066 | 84.48 | 51.9 |

## Iso-time fairness check

| Comparison | Wall budget (s) | Test acc at budget | Final test acc |
| :--------- | --------------: | -----------------: | -------------: |
| TT-DMRG (reference) | 20.77 | **0.8667** | 0.8667 |
| Dense Adam-MSE      | 20.77 | **0.9639** | 0.9806 |
| Dense Adam-CE       | 20.77 | **0.9639** | 0.9667 |
| Large Dense-CE      | 20.77 | **0.9528** | 0.9611 |