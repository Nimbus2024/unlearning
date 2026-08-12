method:
$L = L_{mul}+\gamma L_{uni}+\lambda L_{ret}$
$L_{modal} = E_{x\in D_{modal}}[f(x)]$，$f(x)$ 为任意的 unlearn loss，如 DPO/NPO
DPO 形式：$f(x) = -\log\sigma(\beta\cdot (r_{w}-r_{l}))$，其中 $r(x,y) = \log\frac{\pi_{\theta}(x)}{\pi_{ref}(x)}$ 为 DPO 隐式奖励（实现上取 $CE_{ref}-CE_{\theta}$）
$M_{modal}(x) = r_{w}-r_{l}$（batch 内 margin；$w$=idk 拒绝回答，$l$=正确答案）
$M(x) = M_{uni}(x)-M_{mul}(x)$
$M_{0} = E_{x}[M(x)]$（实现为 EMA：$M_0 \leftarrow \rho M_0 + (1-\rho)M$）
$\gamma = \max(0, (1-\alpha(M-M_{0}))\gamma_{0})$
