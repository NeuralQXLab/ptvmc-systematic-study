---
layout: default
title: My Page with Math
---

<p align="center">

| Scheme | Order | Substeps | Complexity | $U_k$ | $V_k$ | $D_k$ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `LPE-o` | o | o | $\mathcal{O}(N)$ | $1 + a_k \Lambda$ | $\mathbb{1}$ | $\mathbb{1}$ |
| `PPE-o` | o | o/2 | $\mathcal{O}(2N)$ | $1 + a_k \Lambda$ | $1 + b_k \Lambda$ | $\mathbb{1}$ |
| `S-LPE-o` | o | $\dagger$ | $\mathcal{O}(N)$ | $1 + a_k \Lambda_x$ | $\mathbb{1}$ | $\text{exp}(\alpha_k \Lambda_z)$ |
| `S-PPE-o` | o | $\dagger$ | $\mathcal{O}(2N)$ | $1 + a_k \Lambda_x$ | $1 + b_k \Lambda$ | $\text{exp}(\alpha_k \Lambda_z)$ |

</p>
