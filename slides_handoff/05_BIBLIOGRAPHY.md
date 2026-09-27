# Bibliography

For the closing bibliography slide, and for the small source lines at the bottom-right
of each content slide. The numbering used throughout files 02, 03 and 04 is this one.
If you renumber per slide, keep each slide's own source line consistent.

Split across two slides if it does not fit on one.

---

1. R. Ramakrishnan, P. Dral, M. Rupp, O. von Lilienfeld. Quantum chemistry structures
   and properties of 134 kilo molecules. *Scientific Data*, 2014. **(QM9)**
2. E. Hoogeboom, V. Satorras, C. Vignac, M. Welling. Equivariant Diffusion for Molecule
   Generation in 3D. *ICML*, 2022. **(EDM; our stability and validity conventions)**
3. T. Chen, Y. Zhang, S. Tang, P. Chatterjee. Multi-Objective-Guided Discrete Flow
   Matching for Controllable Biological Sequence Design. arXiv:2505.07086, 2025.
   **(MOG-DFM; the enhancer data)**
4. H. Stark, B. Jing, C. Wang, G. Corso, B. Berger, R. Barzilay, T. Jaakkola. Dirichlet
   Flow Matching with Applications to DNA Sequence Design. arXiv:2402.05841, 2024.
5. O. Davis, S. Kessler, M. Petrache, I. Ceylan, M. Bronstein, A. Bose. Fisher Flow
   Matching for Generative Modeling over Discrete Data. *NeurIPS*, 2024.
6. Y. Lipman, R. Chen, H. Ben-Hamu, M. Nickel, M. Le. Flow Matching for Generative
   Modeling. arXiv:2210.02747, 2022.
7. X. Liu, C. Gong, Q. Liu. Flow Straight and Fast: Learning to Generate and Transfer
   Data with Rectified Flow. arXiv:2209.03003, 2022.
8. M. Albergo, N. Boffi, E. Vanden-Eijnden. Stochastic Interpolants: A Unifying
   Framework for Flows and Diffusions. *JMLR*, 2025.
9. V. Satorras, E. Hoogeboom, M. Welling. E(n) Equivariant Graph Neural Networks.
   *ICML*, 2021. **(our backbone)**
10. Y. Song, J. Sohl-Dickstein, D. Kingma, A. Kumar, S. Ermon, B. Poole. Score-Based
    Generative Modeling through Stochastic Differential Equations. *ICLR*, 2021.
11. T. Karras, M. Aittala, T. Aila, S. Laine. Elucidating the Design Space of
    Diffusion-Based Generative Models. *NeurIPS*, 2022.
12. H. Chung, J. Kim, M. McCann, M. Klasky, J. Ye. Diffusion Posterior Sampling for
    General Noisy Inverse Problems. *ICLR*, 2023. **(DPS; our plug-in arm's ancestor)**
13. Velocity-relative trust region: adopted practice, described on slide 2b. Not
    claimed as ours.
14. J. Ho, T. Salimans. Classifier-Free Diffusion Guidance. arXiv:2207.12598, 2022.
15. H. Ye, H. Lin, J. Han, M. Xu, S. Liu, Y. Liang, J. Ma, J. Zou, S. Ermon. TFG:
    Unified Training-Free Guidance for Diffusion Models. *NeurIPS*, 2024.
16. CFG-Ctrl: classifier-free guidance as feedback control. arXiv:2603.03281, 2026.
17. Feedback Guidance of Diffusion Models. arXiv:2506.06085, 2025.
18. Moment Guided Diffusion. arXiv:2602.17211, 2026. **(closest prior art)**
19. Variance-Tilted Diffusion. arXiv:2606.22239, 2026.
20. G. Corso, Y. Xu, V. De Bortoli, R. Barzilay, T. Jaakkola. Particle Guidance:
    Non-I.I.D. Diverse Sampling with Diffusion Models. *ICLR*, 2024.
21. E. Ventura, B. Achilli, L. Ambrogioni, C. Lucibello. Negative guidance reduces means
    and expands variances in diffusion models. arXiv:2602.00716, 2026.
22. B. Boys, M. Girolami, J. Pidstrigach, S. Reich, A. Mosca, O. D. Akyildiz. Tweedie
    Moment Projected Diffusions for Inverse Problems. *TMLR*, 2024. **(TMPD)**
23. J. Song, Q. Zhang, H. Yin, M. Mardani, M.-Y. Liu, J. Kautz, Y. Chen, A. Vahdat.
    Loss-Guided Diffusion Models for Plug-and-Play Controllable Generation. *ICML*, 2023.
    **(LGD-MC)**
24. H. Ben-Hamu, O. Puny, I. Gat, B. Karrer, U. Singer, Y. Lipman. D-Flow:
    Differentiating through Flows for Controlled Generation. *ICML*, 2024.
25. S. Tang, Y. Zhang, A. Tong, P. Chatterjee. Gumbel-Softmax Flow Matching with
    Straight-Through Guidance. arXiv:2503.17361, 2025.

---

## Entries to verify before submitting

Entries **16, 17, 18, 19** are recorded in our repository by arXiv identifier and by
what the method does, but the full author lists were not captured. Check them on arXiv
and complete the names. Everything else is complete.

## Figures

Both figures in `figures/` are **original to this work**. Their source lines say so, as
the rubric requires for any uncited figure. Figure 1's components are drawn after
[6], [2] and [4]; Figure 2 is generated from our own measured cells.
