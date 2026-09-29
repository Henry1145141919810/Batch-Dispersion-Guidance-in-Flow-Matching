import subprocess, json
EQ = {
# 1c-i: paper form of the FM loss
"fm_loss2": r"\mathcal{L}_{\mathrm{FM}}(\theta) = \mathbb{E}\,\big\|\,v_\theta(x_t,t)-(x_1-x_0)\,\big\|_M^2, \qquad \|z\|_M^2 = \frac{\|M\odot z^{c}\|^2}{3\sum M} + \frac{\|M\odot z^{f}\|^2}{5\sum M}",
# 1d-i: paper notation (u clean->noise, x_1 data, eps_phi)
"dm_beta2": r"\beta(u) = \beta_{\min} + u\,(\beta_{\max}-\beta_{\min}) = 0.1 + 19.9\,u",
"dm_fwd2": r"x_u = a(u)\,x_1 + \sigma(u)\,\varepsilon, \qquad a(u) = e^{-\frac12\left(0.1\,u\,+\,9.95\,u^2\right)}, \qquad \sigma^2 = 1-a^2",
"dm_loss": r"\mathcal{L}_{\mathrm{VP}}(\phi) = \mathbb{E}\,\big\|\,\varepsilon_\phi(x_u,u)-\varepsilon\,\big\|_M^2",
"dm_ode2": r"\frac{dx}{du} = -\tfrac12\,\beta(u)\,\big(x + s_\phi(x,u)\big), \qquad s_\phi = -\,\varepsilon_\phi/\sigma",
"dm_end": r"\hat{x}_1 = \big(x_u - \sigma(u)\,\varepsilon_\phi(x_u,u)\big)\,/\,a(u)",
# 2b-i: the plug-in rule and how it enters each sampler
"gd_plug": r"G_i = \frac{y - F_i}{s^2}\;J_i^{\top}\,\nabla_{\!m} f_A(m_i), \qquad F_i = f_A(m_i), \qquad J_i = \frac{\partial m_i}{\partial x^{(i)}}",
"gd_apply": r"\begin{aligned}\text{flow:}\quad \frac{dx}{dt} &= v_\theta + \operatorname{clip}_{\|v_\theta\|}\!\Big(w\,\tfrac{1-t}{t}\,G_i\Big)\\[6pt] \text{diffusion:}\quad \frac{dx}{du} &= f_\phi + \operatorname{clip}_{\|f_\phi\|}\!\Big(\!-\tfrac12\,\beta(u)\,w\,G_i\Big), \qquad f_\phi = -\tfrac12\,\beta(u)\,(x+s_\phi)\end{aligned}",
# 2c-i: the coupling that motivates BDG
"coupling": r"w\,(y - F_i) \;=\; \underbrace{w\,(y-\bar F)}_{\text{moves the batch mean}} \;-\; \underbrace{w\,(F_i-\bar F)}_{\text{shrinks the spread}}",
# 2c-iii: the paper's statements
"red_fact2": r"\begin{aligned} n_i &= (y-F_i) - \eta\,e\,(F_i-\bar F) \;=\; (y-\bar F) - w_{\mathrm{eff}}\,(F_i-\bar F)\\[4pt] &= w_{\mathrm{eff}}\,\big(y_{\mathrm{eff}} - F_i\big), \qquad y_{\mathrm{eff}} = \frac{y+\eta\,e\,\bar F}{w_{\mathrm{eff}}} \qquad (w_{\mathrm{eff}}\neq 0)\end{aligned}",
"red_regime": r"w_{\mathrm{eff}} < 0 \;\iff\; \eta > 1 \ \text{ and } \ V_b < \tau^2\,(1-1/\eta)",
"red_ode": r"\dot F_i = (y-\bar F) - w_{\mathrm{eff}}\,(F_i-\bar F) \;\;\Rightarrow\;\; \dot V_b = -2\,w_{\mathrm{eff}}\,V_b \;\;\Rightarrow\;\; V^{\ast} = \tau^2\,(1-1/\eta)\quad(\eta>1)",
# 3a: the analytic sequence properties
"m2_props": r"F_{\mathrm{GC}}(p) = \frac{1}{L}\sum_{j=1}^{L}\big(p_{j,\mathrm{C}}+p_{j,\mathrm{G}}\big), \qquad F_{\mathrm{CpG}}(p) = \frac{1}{L-1}\sum_{j=1}^{L-1} p_{j,\mathrm{C}}\;p_{j+1,\mathrm{G}}",
}
out = {}
for k, body in EQ.items():
    tex = r"""\documentclass[border=2pt]{standalone}
\usepackage{amsmath,amssymb,xcolor}
\definecolor{ink}{HTML}{16202C}\definecolor{innov}{HTML}{B3261E}
\begin{document}\color{ink}$\displaystyle %s$\end{document}""" % body
    open(k+".tex","w").write(tex)
    r = subprocess.run(["pdflatex","-interaction=nonstopmode",k+".tex"],capture_output=True,text=True)
    if r.returncode: print("FAIL",k, r.stdout[-900:]); continue
    subprocess.run(["pdftocairo","-png","-transp","-r","518.4","-singlefile",k+".pdf","eq_"+k],check=True)
    info = subprocess.run(["pdfinfo",k+".pdf"],capture_output=True,text=True).stdout
    line=[l for l in info.splitlines() if l.startswith("Page size")][0]
    w,h = float(line.split()[2]), float(line.split()[4])
    out[k]=(round(w*3.6), round(h*3.6))
    print(k, out[k])
json.dump(out, open("dims.json","w"))
