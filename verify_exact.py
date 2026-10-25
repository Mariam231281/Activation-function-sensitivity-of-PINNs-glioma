"""Symbolic check that each exact solution satisfies its PDE (Appendix A)."""
import sympy as sp
x, t = sp.symbols("x t")
cases = {
    "Example 1": (sp.exp(x + t), lambda u: u / 2),
    "Example 2": (sp.log(x + t + 2), lambda u: sp.exp(-u) + sp.exp(-2 * u) / 2),
    "Example 3": (sp.exp(2 * x + 2 * t - t**3 / 3), lambda u: -t**2 * u),
    "Example 3 (original, inconsistent)": (sp.exp(2 * x + 2 * t - t**3 / 3), lambda u: t**2 * u),
}
for name, (u, F) in cases.items():
    r = sp.simplify(sp.diff(u, t) - sp.Rational(1, 2) * sp.diff(u, x, 2) - F(u))
    print(f"{name}: residual = {sp.factor(r)}")
