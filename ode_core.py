"""ส่วนคำนวณทางคณิตศาสตร์ (ไม่ผูกกับ Streamlit/Plotly) ใช้ร่วมกับ app.py"""
from dataclasses import dataclass

import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

# ---------------------------------------------------------------- Math parser
X, Y, A, B, C = sp.symbols("x y a b c")
TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)
LOCALS = {"x": X, "y": Y, "a": A, "b": B, "c": C, "pi": sp.pi}


def parse_f(text):
    """แปลงข้อความสมการ f(x, y, a, b, c) -> (sympy expr, numpy function)"""
    expr = parse_expr(text, local_dict=LOCALS, transformations=TRANSFORMS)
    expr = sp.sympify(expr)
    extra = expr.free_symbols - {X, Y, A, B, C}
    if extra:
        raise ValueError(f"ตัวแปรที่ไม่รู้จัก: {', '.join(map(str, extra))} (ใช้ได้เฉพาะ x, y, a, b, c)")
    return expr, _numpy_fn(expr)


def _numpy_fn(expr):
    """lambdify แล้วห่อให้รับ/คืนชนิด numpy เสมอ (หารศูนย์/ล้น -> inf/nan แทนที่จะ raise)"""
    raw = sp.lambdify((X, Y, A, B, C), expr, modules=["numpy"])

    def fn(x, y, a, b, c):
        return raw(*(np.asarray(v, dtype=np.float64) for v in (x, y, a, b, c)))

    return fn


def make_fy(expr):
    """∂f/∂y เป็นทั้งนิพจน์และฟังก์ชัน numpy"""
    fy_expr = sp.diff(expr, Y)
    return fy_expr, _numpy_fn(fy_expr)


# ------------------------------------------------------------- Exact solutions
def exact_cooling(x, x0, y0, a, b, c):
    return b + (y0 - b) * np.exp(-a * (x - x0))


def exact_logistic(x, x0, y0, a, b, c):
    if y0 == 0:
        return np.zeros_like(x, dtype=float)
    return b / (1 + ((b - y0) / y0) * np.exp(-a * (x - x0)))


def exact_tank(x, x0, y0, a, b, c):
    q_inf = a * c
    return q_inf + (y0 - q_inf) * np.exp(-(b / c) * (x - x0))


# ------------------------------------------------------------------- Solvers
def euler(fn, p, x0, y0, h, n):
    """Euler: h ติดลบได้ (คำนวณย้อนไปทาง x ที่ลดลง)"""
    ys = np.empty(n + 1)
    ys[0] = y0
    with np.errstate(all="ignore"):
        for i in range(n):
            ys[i + 1] = ys[i] + h * fn(x0 + i * h, ys[i], *p)
    return ys


def rk4_step_ks(fn, p, x, y, h):
    k1 = fn(x, y, *p)
    k2 = fn(x + h / 2, y + h * k1 / 2, *p)
    k3 = fn(x + h / 2, y + h * k2 / 2, *p)
    k4 = fn(x + h, y + h * k3, *p)
    return k1, k2, k3, k4


def rk4(fn, p, x0, y0, h, n):
    ys = np.empty(n + 1)
    ys[0] = y0
    with np.errstate(all="ignore"):
        for i in range(n):
            k1, k2, k3, k4 = rk4_step_ks(fn, p, x0 + i * h, ys[i], h)
            ys[i + 1] = ys[i] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return ys


def solve_range(method, fn, p, x0, y0, h, n_back, n_fwd):
    """คำตอบบน x0 - n_back*h ... x0 + n_fwd*h (รวมทั้งย้อนหลังและไปข้างหน้า)"""
    fwd = method(fn, p, x0, y0, h, n_fwd)
    bwd = method(fn, p, x0, y0, -h, n_back)
    return np.concatenate([bwd[:0:-1], fwd])


def curve(fn, p, x0, yi, x_start, x_end, npts=400):
    """เส้นคำตอบ (RK4 ก้าวเล็ก) สำหรับวาดตระกูลเส้นคำตอบ"""
    hf = (x_end - x_start) / npts
    nb = int(round((x0 - x_start) / hf))
    nf = int(round((x_end - x0) / hf))
    ys = solve_range(rk4, fn, p, x0, yi, hf, nb, nf)
    xs = x0 + hf * np.arange(-nb, nf + 1)
    return xs, ys


# ---------------------------------------------------- Symbolic exact solution
def symbolic_exact(f_text, a, b, c, x0, y0):
    """พยายามหาคำตอบรูปปิดด้วย SymPy (เฉพาะสมการเชิงเส้น/แยกแปรได้/Bernoulli เพื่อไม่ให้ค้าง)
    คืนค่าเป็นสตริงของ y(x) หรือ None"""
    try:
        expr, _ = parse_f(f_text)
        sub = {A: sp.nsimplify(a), B: sp.nsimplify(b), C: sp.nsimplify(c)}
        yf = sp.Function("yf")
        rhs = expr.subs(sub).subs(Y, yf(X))
        ode = sp.Eq(yf(X).diff(X), rhs)
        ics = {yf(sp.nsimplify(x0)): sp.nsimplify(y0)}
        for hint in ("1st_linear", "separable", "Bernoulli"):
            try:
                sol = sp.dsolve(ode, yf(X), hint=hint, ics=ics)
            except Exception:
                continue
            for s in (sol if isinstance(sol, list) else [sol]):
                if s.lhs == yf(X) and not s.rhs.has(yf):
                    return str(s.rhs)
    except Exception:
        pass
    return None


def eval_symbolic(sol_str, xs, y_ref, y0, x0):
    """แปลงสตริงคำตอบเป็นค่าตัวเลข และตรวจความสมเหตุสมผลเทียบกับคำตอบอ้างอิง (RK4 ก้าวเล็ก)
    คืน (expr, ys) หรือ (None, None) ถ้าตรวจไม่ผ่าน"""
    try:
        e = sp.sympify(sol_str, locals={"x": X})
        f = sp.lambdify(X, e, modules=["numpy"])
        with np.errstate(all="ignore"):
            ys = np.asarray(f(xs), dtype=float) + 0 * xs
            y_at_x0 = float(f(x0))
        ok = np.isfinite(ys).all() and abs(y_at_x0 - y0) < 1e-6 * (1 + abs(y0))
        scale = 1 + np.nanmax(np.abs(y_ref[np.isfinite(y_ref)]))
        ok = ok and np.nanmax(np.abs(ys - y_ref)) < 1e-3 * scale
        return (e, ys) if ok else (None, None)
    except Exception:
        return None, None


# --------------------------------------------------- Coordinate mapping / field
@dataclass
class CoordMap:
    """แปลงพิกัดคณิตศาสตร์ (x, y) <-> พิกเซลของ canvas (px, py) โดย py ชี้ลงล่าง"""
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    W: int = 900
    H: int = 560

    def to_px(self, x, y):
        px = (x - self.xmin) / (self.xmax - self.xmin) * self.W
        py = self.H - (y - self.ymin) / (self.ymax - self.ymin) * self.H
        return px, py

    def to_math(self, px, py):
        x = self.xmin + px / self.W * (self.xmax - self.xmin)
        y = self.ymin + (self.H - py) / self.H * (self.ymax - self.ymin)
        return x, y

    def slope_to_pixel_angle(self, s):
        """ความชัน dy/dx (คณิตศาสตร์) -> มุมของขีดบนจอ (เรเดียน, แกน py ชี้ลง)"""
        k = (self.H * (self.xmax - self.xmin)) / (self.W * (self.ymax - self.ymin))
        return np.arctan2(-s * k, 1.0)


def slope_field_segments(fn, p, cmap, nx, ny):
    """สร้างขีดสโลปฟิลด์ โดยคำนวณทิศทางใน 'พิกเซลสเปซ' แล้วแปลงกลับเป็นพิกัดคณิตศาสตร์
    คืน (xs_line, ys_line) แบบมี NaN คั่นระหว่างขีด สำหรับ plot ทีเดียว"""
    gx = np.linspace(cmap.xmin, cmap.xmax, nx)
    gy = np.linspace(cmap.ymin, cmap.ymax, ny)
    XX, YY = np.meshgrid(gx, gy)
    with np.errstate(all="ignore"):
        S = np.asarray(fn(XX, YY, *p), dtype=float) + 0 * XX
    PX, PY = cmap.to_px(XX, YY)
    phi = cmap.slope_to_pixel_angle(S)
    half = 0.4 * min(cmap.W / nx, cmap.H / ny)
    x1, y1 = cmap.to_math(PX - half * np.cos(phi), PY - half * np.sin(phi))
    x2, y2 = cmap.to_math(PX + half * np.cos(phi), PY + half * np.sin(phi))
    gap = np.full_like(XX, np.nan)
    lx = np.stack([x1, x2, gap], axis=-1).ravel()
    ly = np.stack([y1, y2, gap], axis=-1).ravel()
    return lx, ly


# --------------------------------------------------------- Equilibrium analysis
def is_autonomous(expr):
    return X not in expr.free_symbols


def find_equilibria(fn, p, x0, ymin, ymax):
    """หาคำตอบของ f(y) = 0 บน [ymin, ymax] (สมการออโตโนมัส) ด้วยการสแกนเครื่องหมายและ bisection"""
    ys = np.linspace(ymin, ymax, 4001)
    with np.errstate(all="ignore"):
        v = np.asarray(fn(x0, ys, *p), dtype=float) + 0 * ys
        finite = v[np.isfinite(v)]
        tol = 1e-6 * max(1.0, float(np.max(np.abs(finite))) if finite.size else 1.0)

        def g(y):
            return float(fn(x0, y, *p))

        roots = []
        for i in range(len(ys) - 1):
            if not (np.isfinite(v[i]) and np.isfinite(v[i + 1])):
                continue
            if v[i] == 0:
                roots.append(float(ys[i]))
            elif v[i] * v[i + 1] < 0:
                lo, hi = float(ys[i]), float(ys[i + 1])
                for _ in range(60):
                    mid = (lo + hi) / 2
                    gm = g(mid)
                    if gm == 0:
                        lo = hi = mid
                        break
                    if g(lo) * gm < 0:
                        hi = mid
                    else:
                        lo = mid
                r = (lo + hi) / 2
                if abs(g(r)) <= tol:  # ตัดจุดที่เป็นแค่การข้ามเสา (pole) ออก
                    roots.append(r)
        if np.isfinite(v[-1]) and v[-1] == 0:
            roots.append(float(ys[-1]))
    snap = 1e-9 * max(1.0, abs(ymax - ymin))
    roots = [0.0 if abs(r) < snap else r for r in roots]
    out = []
    for r in sorted(roots):
        if not out or abs(r - out[-1]) > 1e-6 * max(1.0, abs(r)):
            out.append(r)
    return out


def classify_equilibrium(fn, p, x0, r, d):
    with np.errstate(all="ignore"):
        left, right = float(fn(x0, r - d, *p)), float(fn(x0, r + d, *p))
    if not (np.isfinite(left) and np.isfinite(right)):
        return "ระบุไม่ได้"
    if left > 0 > right:
        return "เสถียร (Stable)"
    if left < 0 < right:
        return "ไม่เสถียร (Unstable)"
    return "กึ่งเสถียร (Semi-stable)"


# ---------------------------------------------- Existence / uniqueness checking
def grid_check(fn, fy_fn, p, xlo, xhi, ylo, yhi, m=41):
    """ตรวจเชิงตัวเลขบนตารางจุดในสี่เหลี่ยม: สัดส่วนที่ f, ∂f/∂y มีค่าจำกัด, M=max|f|, L=max|∂f/∂y|"""
    gx = np.linspace(xlo, xhi, m)
    gy = np.linspace(ylo, yhi, m)
    GX, GY = np.meshgrid(gx, gy)
    with np.errstate(all="ignore"):
        F = np.asarray(fn(GX, GY, *p), dtype=float) + 0 * GX
        FY = np.asarray(fy_fn(GX, GY, *p), dtype=float) + 0 * GX
    ff, fyf = np.isfinite(F), np.isfinite(FY)
    M = float(np.max(np.abs(F[ff]))) if ff.any() else float("nan")
    L = float(np.max(np.abs(FY[fyf]))) if fyf.any() else float("nan")
    return float(ff.mean()), float(fyf.mean()), M, L


def denominators(expr):
    num, den = sp.fraction(sp.together(expr))
    return den
