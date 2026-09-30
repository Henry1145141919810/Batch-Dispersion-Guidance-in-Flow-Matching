from pathlib import Path
import re

root = Path(__file__).resolve().parent.parent
p = root / 'main.tex'
s = p.read_text(encoding='utf-8')
s = s.replace(r'\begin{table}[t]', r'\begin{table}[htbp]')
s = s.replace(r'\begin{table}[h]', r'\begin{table}[htbp]')
s = s.replace(r'\begin{figure}[t]', r'\begin{figure}[htbp]')
s = s.replace(r'\begin{algorithm}[h]', r'\begin{algorithm}[htbp]')
s = s.replace('Continuous generative models transform simple reference distributions into structured samples. Diffusion learns to reverse a corruption process, and flow matching learns a velocity field along a specified probability path', 'Diffusion reverses a corruption process, and flow matching learns transport along a prescribed probability path')
s = s.replace('A matched comparison can identify which family provides a stronger substrate for controlled molecular generation.', 'We compare their suitability for controlled molecular generation.')
s = s.replace('Gradient guidance steers a frozen generator toward a property target', 'Gradient guidance steers a frozen generator toward a target')
s = s.replace('For a quadratic property energy, its strength scales both target centering and contraction of within-batch property deviations. This coupling motivates a separate coefficient for dispersion.', 'Its quadratic property energy couples target centering and contraction, motivating a separate dispersion coefficient.')
s = s.replace('The design compares linear flow matching with variance-preserving diffusion, tests plug-in guidance, isolates BDG through ablations, and reserves recent-method comparisons in each modality.', 'We compare linear FM and VP diffusion, test plug-in guidance, ablate BDG, and plan recent-method benchmarks.')
s = s.replace('We distinguish dispersion gain, setpoint, strength, and feedback through controlled comparisons.', 'We specify controls for dispersion gain, setpoint, strength, and feedback.')
p.write_text(s, encoding='utf-8')
p = root / 'refs_extra.bib'
s = p.read_text(encoding='utf-8')
# BibTeX does not accept percent comments between fields inside an entry.
s = '\n'.join(line for line in s.splitlines() if not line.lstrip().startswith('%')) + '\n'
p.write_text(s, encoding='utf-8')
p = root / 'citation.bib'
s = p.read_text(encoding='utf-8')
s = re.sub(r'@article\{stark2024dirichlet,.*?\n\}', r'''@inproceedings{stark2024dirichlet,
  title={Dirichlet Flow Matching with Applications to {DNA} Sequence Design},
  author={Stark, Hannes and Jing, Bowen and Wang, Chenyu and Corso, Gabriele and Berger, Bonnie and Barzilay, Regina and Jaakkola, Tommi},
  booktitle={Proceedings of the 41st International Conference on Machine Learning},
  volume={235},
  pages={46495--46513},
  year={2024},
  publisher={PMLR},
  url={https://proceedings.mlr.press/v235/stark24b.html}
}''', s, flags=re.S)
p.write_text(s, encoding='utf-8')
