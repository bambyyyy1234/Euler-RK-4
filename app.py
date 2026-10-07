import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import sympy as sp

import ode_core as oc

st.set_page_config(page_title="Slope Field & Euler/RK4 Solver", layout="wide")


@st.cache_data(show_spinner=False)
def cached_symbolic(f_text, a, b, c, x0, y0):
    return oc.symbolic_exact(f_text, a, b, c, x0, y0)


def fmt(v):
    return f"{float(v):.6g}"


def par(v):
    return f"({fmt(v)})" if float(v) < 0 else fmt(v)


def safe_max(arr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    return float(arr.max()) if arr.size else float("nan")


# ------------------------------------------------------------------- Presets
PRESETS = {
    "กำหนดเอง (Custom)": dict(
        f="x - y", labels=("a", "b", "c"), params=(1.0, 1.0, 1.0),
        x_start=-2.0, x0=0.0, y0=1.0, x_end=5.0, h=0.5, exact=None,
        note="สมการที่ผู้ใช้กำหนดเอง: ใช้ x, y และพารามิเตอร์ a, b, c ได้ "
             "แอปจะลองหาคำตอบรูปปิดด้วย SymPy (เชิงเส้น / แยกแปรได้ / Bernoulli) "
             "ถ้าหาไม่ได้จะใช้ RK4 ก้าวเล็กมากเป็นคำตอบอ้างอิง",
    ),
    "กฎการเย็นตัวของนิวตัน": dict(
        f="-a*(y - b)", labels=("k", "T_m", "-"), params=(1.0, 20.0, 1.0),
        x_start=0.0, x0=0.0, y0=90.0, x_end=5.0, h=0.5, exact=oc.exact_cooling,
        note="dT/dt = -k(T - Tm): ที่ T = Tm ความชันเป็น 0 (จุดสมดุลเสถียร) "
             "ถ้า T > Tm ความชันติดลบ, ถ้า T < Tm ความชันเป็นบวก "
             "ลองเพิ่ม k มาก ๆ แล้วใช้ h ใหญ่ จะเห็น Euler แกว่ง/หลุดลอย (h·k > 2) ขณะที่ RK4 ยังแม่นยำ",
    ),
    "โมเดลประชากรโลจิสติก": dict(
        f="a*y*(1 - y/b)", labels=("r", "K", "-"), params=(1.0, 100.0, 1.0),
        x_start=0.0, x0=0.0, y0=10.0, x_end=10.0, h=0.5, exact=oc.exact_logistic,
        note="dP/dt = rP(1 - P/K): จุดสมดุล P = 0 (ไม่เสถียร) และ P = K (เสถียร) "
             "เส้นคำตอบเป็นรูปตัว S เมื่อ 0 < P < K และมีจุดเปลี่ยนความเว้าที่ P = K/2",
    ),
    "ปัญหาการผสมสารในถัง": dict(
        f="a*b - (b/c)*y", labels=("c_in", "r", "V0"), params=(0.5, 5.0, 100.0),
        x_start=0.0, x0=0.0, y0=0.0, x_end=60.0, h=2.0, exact=oc.exact_tank,
        note="dQ/dt = c_in·r - (r/V0)Q: เมื่อ t → ∞ ปริมาณสาร Q ลู่เข้าสู่ค่าคงตัว "
             "Q∞ = c_in·V0 ไม่ว่า Q(0) จะเริ่มที่เท่าใด",
    ),
}

# ------------------------------------------------------------------- Sidebar
st.title("Interactive Slope Field & Euler/RK4 ODE Solver")
st.caption("dy/dx = f(x, y), y(x₀) = y₀  |  เปรียบเทียบวิธีออยเลอร์ กับ รุงเงอ-คุตตาอันดับ 4")

st.sidebar.header("ตั้งค่า")
preset_name = st.sidebar.selectbox("ตัวอย่าง (Preset)", list(PRESETS.keys()))
P = PRESETS[preset_name]
k = preset_name  # ใช้เป็นส่วนหนึ่งของ key เพื่อให้ค่าเริ่มต้นรีเซ็ตเมื่อเปลี่ยน preset
is_custom = P["exact"] is None

f_text = st.sidebar.text_input("f(x, y) =", value=P["f"], key=f"f_{k}", disabled=not is_custom,
                               help="ตัวอย่าง: sin(x) - y, a*y*(1-y/b), x**2 + y, exp(-x)*y")

st.sidebar.subheader("พารามิเตอร์ของสมการ")
params = []
for name, label, default in zip("abc", P["labels"], P["params"]):
    if label == "-":
        params.append(default)
        continue
    params.append(st.sidebar.number_input(f"{label}  (ตัวแปร {name})", value=float(default),
                                          step=0.1, format="%.4f", key=f"{name}_{k}"))
a, b, c = params

st.sidebar.subheader("เงื่อนไขเริ่มต้น และช่วงคำนวณ")
x_start = st.sidebar.number_input("x_start", value=float(P["x_start"]), step=0.5, key=f"xs_{k}")
x0 = st.sidebar.number_input("x₀", value=float(P["x0"]), step=0.5, key=f"x0_{k}")
y0 = st.sidebar.number_input("y₀", value=float(P["y0"]), step=0.5, key=f"y0_{k}")
x_end = st.sidebar.number_input("x_end", value=float(P["x_end"]), step=0.5, key=f"xe_{k}")
h = st.sidebar.number_input("ขนาดก้าว h", value=float(P["h"]), min_value=0.0001, step=0.05,
                            format="%.4f", key=f"h_{k}")

st.sidebar.subheader("การแสดงผล")
show_euler = st.sidebar.checkbox("Euler", value=True)
show_rk4 = st.sidebar.checkbox("RK4", value=True)
show_exact = st.sidebar.checkbox("Exact / Reference", value=True)
show_eq = st.sidebar.checkbox("แสดงจุดสมดุล (สมการออโตโนมัส)", value=True)
density = st.sidebar.slider("ความหนาแน่นของ Slope Field", 8, 40, 22)
auto_y = st.sidebar.checkbox("ปรับช่วงแกน y อัตโนมัติ", value=True)

st.sidebar.subheader("ตระกูลเส้นคำตอบ")
show_family = st.sidebar.checkbox("แสดงเส้นคำตอบจากหลายเงื่อนไขเริ่มต้น", value=False)
n_family = st.sidebar.slider("จำนวนเส้นอัตโนมัติ", 3, 15, 7, disabled=not show_family)
extra_text = st.sidebar.text_input("y₀ เพิ่มเติม (คั่นด้วย ,)", value="", disabled=not show_family,
                                   help="เช่น -2, 0, 2.5 (เส้นทั้งหมดเริ่มที่ x = x₀)")

# --------------------------------------------------------------- Validation
if not (x_start <= x0 < x_end):
    st.error("ต้องให้ x_start ≤ x₀ < x_end")
    st.stop()

n_b = int(round((x0 - x_start) / h))
n_f = max(1, int(round((x_end - x0) / h)))
if n_b + n_f > 5000:
    st.error(f"จำนวนก้าว {n_b + n_f} มากเกินไป (สูงสุด 5000) กรุณาเพิ่ม h")
    st.stop()

try:
    expr, fn = oc.parse_f(f_text)
    fy_expr, fy_fn = oc.make_fy(expr)
except Exception as e:
    st.error(f"อ่านสมการไม่ได้: {e}")
    st.stop()

st.latex(r"\frac{dy}{dx} = " + sp.latex(expr))

# -------------------------------------------------------------- Computation
p = tuple(np.float64(v) for v in (a, b, c))
n_total = n_b + n_f
exact_expr = None
try:
    with np.errstate(all="ignore"):
        xs = x0 + h * np.arange(-n_b, n_f + 1)
        y_euler = oc.solve_range(oc.euler, fn, p, x0, y0, h, n_b, n_f)
        y_rk4 = oc.solve_range(oc.rk4, fn, p, x0, y0, h, n_b, n_f)
        if not is_custom:
            y_exact = P["exact"](xs, x0, y0, *p)
            ref_label = "Exact"
        else:
            sub = max(2, min(20, 40000 // n_total))
            y_ref = oc.solve_range(oc.rk4, fn, p, x0, y0, h / sub, n_b * sub, n_f * sub)[::sub]
            sol_str = cached_symbolic(f_text, a, b, c, x0, y0)
            exact_expr, y_sym = (oc.eval_symbolic(sol_str, xs, y_ref, y0, x0)
                                 if sol_str else (None, None))
            if exact_expr is not None:
                y_exact, ref_label = y_sym, "Exact (SymPy)"
            else:
                y_exact, ref_label = y_ref, f"Reference (RK4, h/{sub})"
except (ZeroDivisionError, OverflowError, ValueError, FloatingPointError) as e:
    st.error(f"คำนวณไม่ได้ด้วยค่าที่กำหนด: {e}")
    st.stop()

with np.errstate(all="ignore"):
    err_euler = np.abs(y_exact - y_euler)
    err_rk4 = np.abs(y_exact - y_rk4)

if exact_expr is not None:
    st.latex(r"y(x) = " + sp.latex(sp.simplify(exact_expr)))
elif is_custom:
    st.caption("หาคำตอบรูปปิดด้วย SymPy ไม่ได้ จึงใช้ RK4 ก้าวเล็กเป็นคำตอบอ้างอิง")

# ------------------------------------------------------------- y-range & misc
if auto_y:
    finite = np.concatenate([y_exact[np.isfinite(y_exact)], y_rk4[np.isfinite(y_rk4)], [y0]])
    limit = 1e4 * (1 + abs(y0))
    if np.abs(finite).max() > limit:
        # คำตอบพุ่งเกินขอบเขต (blow-up): ตัดค่าที่ใหญ่เกินไปออก เพื่อให้ยังเห็นสนามทิศทาง
        finite = finite[np.abs(finite) <= limit]
        st.caption("คำตอบพุ่งเกินขอบเขต (blow-up) จึงตัดช่วงแกน y อัตโนมัติ")
    lo, hi = float(finite.min()), float(finite.max())
    pad = 0.3 * (hi - lo) if hi - lo > 1e-9 else 1.0
    ymin, ymax = lo - pad, hi + pad
else:
    ymin = st.sidebar.number_input("y min", value=-5.0)
    ymax = st.sidebar.number_input("y max", value=5.0)
    if ymax <= ymin:
        st.error("ต้องให้ y max > y min")
        st.stop()

cmap = oc.CoordMap(x_start, x_end, ymin, ymax)  # แปลงพิกัดคณิตศาสตร์ <-> พิกเซล
autonomous = oc.is_autonomous(expr)
equilibria = oc.find_equilibria(fn, p, x0, ymin, ymax) if autonomous else []

family_y0 = []
if show_family:
    family_y0 = [float(v) for v in np.linspace(ymin, ymax, n_family)]
    for tok in extra_text.split(","):
        tok = tok.strip()
        if tok:
            try:
                family_y0.append(float(tok))
            except ValueError:
                st.sidebar.warning(f"ข้ามค่า '{tok}' (ไม่ใช่ตัวเลข)")

# -------------------------------------------------------------------- Tabs
tab_plot, tab_table, tab_err, tab_step, tab_eq, tab_uniq = st.tabs([
    "กราฟ Slope Field", "ตารางผลลัพธ์", "ค่าคลาดเคลื่อน", "ทีละขั้น",
    "จุดสมดุล/ทฤษฎี", "การมีอยู่และเอกลักษณ์"
    ])

# ---- 1) กราฟ
with tab_plot:
    fig = go.Figure()
    lx, ly = oc.slope_field_segments(fn, p, cmap, density, density)
    fig.add_trace(go.Scatter(x=lx, y=ly, mode="lines", name="Slope Field", hoverinfo="skip",
                             line=dict(color="rgba(120,120,120,0.7)", width=1.5)))

    for j, yi in enumerate(family_y0):
        with np.errstate(all="ignore"):
            fx, fyv = oc.curve(fn, p, x0, yi, x_start, x_end)
        fig.add_trace(go.Scatter(x=fx, y=fyv, mode="lines", legendgroup="family",
                                 name="ตระกูลเส้นคำตอบ", showlegend=(j == 0),
                                 line=dict(color="rgba(31,119,180,0.45)", width=1.5),
                                 hovertemplate=f"y(x₀)={yi:.4g}<extra></extra>"))

    if show_eq and autonomous:
        for r in equilibria:
            kind = oc.classify_equilibrium(fn, p, x0, r, 1e-3 * (ymax - ymin))
            col = "seagreen" if kind.startswith("เสถียร") else (
                "darkorange" if kind.startswith("ไม่") else "gray")
            fig.add_hline(y=r, line_dash="dot", line_color=col,
                          annotation_text=f"y = {r:.4g} · {kind}", annotation_position="top left")

    mk = dict(size=6) if n_total <= 60 else dict(size=0)
    if show_exact:
        fig.add_trace(go.Scatter(x=xs, y=y_exact, mode="lines", name=ref_label,
                                 line=dict(color="black", width=3)))
    if show_euler:
        fig.add_trace(go.Scatter(x=xs, y=y_euler, mode="lines+markers", name="Euler",
                                 line=dict(color="crimson", width=2), marker=mk))
    if show_rk4:
        fig.add_trace(go.Scatter(x=xs, y=y_rk4, mode="lines+markers", name="RK4",
                                 line=dict(color="royalblue", width=2, dash="dash"), marker=mk))
    fig.add_trace(go.Scatter(x=[x0], y=[y0], mode="markers", name="(x₀, y₀)",
                             marker=dict(color="green", size=11, symbol="diamond")))
    fig.update_layout(height=620, xaxis=dict(title="x", range=[x_start, x_end]),
                      yaxis=dict(title="y", range=[ymin, ymax]),
                      legend=dict(orientation="h", y=1.08), margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig)
    st.caption(f"ก้าวย้อนหลัง {n_b} ก้าว (x₀ → x_start) และก้าวไปข้างหน้า {n_f} ก้าว (x₀ → x_end), h = {h:g}")

# ---- 2) ตาราง
with tab_table:
    df = pd.DataFrame({
        "n": np.arange(-n_b, n_f + 1), "x_n": xs,
        "y Euler": y_euler, "y RK4": y_rk4, f"y {ref_label}": y_exact,
        "|e| Euler": err_euler, "|e| RK4": err_rk4,
    })
    st.dataframe(df, hide_index=True)
    st.caption("n < 0 คือก้าวย้อนหลังจาก x₀ ไปทาง x_start (ใช้ก้าว −h)")
    st.download_button("ดาวน์โหลด CSV", df.to_csv(index=False).encode("utf-8-sig"),
                       file_name="ode_results.csv", mime="text/csv")

# ---- 3) Error
with tab_err:
    me, mr = safe_max(err_euler), safe_max(err_rk4)
    c1, c2, c3 = st.columns(3)
    c1.metric("Max |e| Euler", f"{me:.3e}")
    c2.metric("Max |e| RK4", f"{mr:.3e}")
    c3.metric("Euler / RK4", f"{me / max(mr, 1e-300):,.1f} เท่า")
    log_y = st.checkbox("แกน y แบบ log", value=True)
    fe = go.Figure()
    fe.add_trace(go.Scatter(x=xs, y=err_euler, name="|e| Euler", line=dict(color="crimson")))
    fe.add_trace(go.Scatter(x=xs, y=err_rk4, name="|e| RK4", line=dict(color="royalblue")))
    fe.update_layout(height=450, xaxis_title="x", yaxis_title="Absolute Error",
                     yaxis_type="log" if log_y else "linear")
    st.plotly_chart(fe)
    st.caption("Euler มีความแม่นยำอันดับ O(h), RK4 มีความแม่นยำอันดับ O(h⁴) "
               "ลองลด h ลงครึ่งหนึ่งแล้วดูว่า error ของแต่ละวิธีลดลงกี่เท่า")

# ---- 4) ทีละขั้น
with tab_step:
    st.markdown("อธิบายการคำนวณทีละก้าว **ไปข้างหน้าจาก x₀** (ค่า yₙ ของแต่ละวิธีใช้ค่าที่วิธีนั้นคำนวณได้เอง)")
    i = st.slider("เลือกก้าวที่ n  (xₙ → xₙ₊₁)", 0, n_f - 1, 0) if n_f > 1 else 0
    idx = n_b + i
    xn = x0 + i * h
    ye, yr_ = float(y_euler[idx]), float(y_rk4[idx])

    with np.errstate(all="ignore"):
        fe_val = float(fn(xn, ye, *p))
        k1, k2, k3, k4 = [float(v) for v in oc.rk4_step_ks(fn, p, xn, yr_, h)]
    ye_next = ye + h * fe_val
    yr_next = yr_ + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    st.markdown("**วิธีของออยเลอร์**")
    st.latex(rf"y_{{{i+1}}} = y_{{{i}}} + h\,f(x_{{{i}}}, y_{{{i}}})"
             rf" = {fmt(ye)} + {fmt(h)}\cdot f({fmt(xn)}, {fmt(ye)})"
             rf" = {fmt(ye)} + {fmt(h)}\cdot {par(fe_val)} = {fmt(ye_next)}")
    if np.isfinite(fe_val):
        trend = ("เป็นบวก → ขีดทิศทางชี้ขึ้น y กำลังเพิ่มขึ้น" if fe_val > 0 else
                 "เป็นลบ → ขีดทิศทางชี้ลง y กำลังลดลง" if fe_val < 0 else
                 "เป็นศูนย์ → ขีดทิศทางแนวนอน (อาจเป็นจุดสมดุลหรือจุดวิกฤต)")
        st.caption(f"ความชัน f(xₙ, yₙ) = {fmt(fe_val)} {trend}")

    st.markdown("**วิธี RK4**")
    st.latex(rf"k_1 = f({fmt(xn)}, {fmt(yr_)}) = {fmt(k1)}")
    st.latex(rf"k_2 = f\left(x_{{{i}}}+\frac{{h}}{{2}},\, y_{{{i}}}+h\frac{{k_1}}{{2}}\right)"
             rf" = f({fmt(xn + h/2)}, {fmt(yr_ + h*k1/2)}) = {fmt(k2)}")
    st.latex(rf"k_3 = f\left(x_{{{i}}}+\frac{{h}}{{2}},\, y_{{{i}}}+h\frac{{k_2}}{{2}}\right)"
             rf" = f({fmt(xn + h/2)}, {fmt(yr_ + h*k2/2)}) = {fmt(k3)}")
    st.latex(rf"k_4 = f\left(x_{{{i}}}+h,\, y_{{{i}}}+h\,k_3\right)"
             rf" = f({fmt(xn + h)}, {fmt(yr_ + h*k3)}) = {fmt(k4)}")
    st.latex(rf"y_{{{i+1}}} = {fmt(yr_)} + \frac{{{fmt(h)}}}{{6}}"
             rf"\left({fmt(k1)} + 2{par(k2)} + 2{par(k3)} + {par(k4)}\right) = {fmt(yr_next)}")

    st.markdown(f"**เทียบกับ {ref_label} ที่ x = {fmt(xn + h)}**")
    cc1, cc2, cc3 = st.columns(3)
    cc1.metric(ref_label, fmt(y_exact[idx + 1]))
    cc2.metric("|e| Euler", f"{err_euler[idx + 1]:.3e}")
    cc3.metric("|e| RK4", f"{err_rk4[idx + 1]:.3e}")

    with st.expander("ตาราง k₁–k₄ ของ RK4 ทุกก้าว (ไปข้างหน้า)"):
        rows = []
        with np.errstate(all="ignore"):
            for j in range(n_f):
                xj, yj = x0 + j * h, float(y_rk4[n_b + j])
                kk = [float(v) for v in oc.rk4_step_ks(fn, p, xj, yj, h)]
                rows.append({"n": j, "x_n": xj, "y_n": yj, "k1": kk[0], "k2": kk[1],
                             "k3": kk[2], "k4": kk[3], "y_n+1": float(y_rk4[n_b + j + 1])})
        st.dataframe(pd.DataFrame(rows), hide_index=True)

# ---- 5) จุดสมดุล / ทฤษฎี
with tab_eq:
    st.markdown(P["note"])
    st.markdown("#### จุดสมดุลและเสถียรภาพ")
    if not autonomous:
        st.info("สมการนี้ขึ้นกับ x (non-autonomous) จึงโดยทั่วไปไม่มีคำตอบสมดุลที่เป็นค่าคงที่ "
                "ให้สังเกตจาก Slope Field และตระกูลเส้นคำตอบแทน")
    elif not equilibria:
        st.info("ไม่พบจุดสมดุล (f(y) = 0) ในช่วงแกน y ที่แสดง")
    else:
        rows = []
        for r in equilibria:
            with np.errstate(all="ignore"):
                slope = float(fy_fn(x0, r, *p))
            rows.append({"จุดสมดุล y*": r,
                         "ชนิด": oc.classify_equilibrium(fn, p, x0, r, 1e-3 * (ymax - ymin)),
                         "∂f/∂y ที่ y*": slope})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption("จำแนกจากเครื่องหมายของ f ทางซ้าย/ขวาของ y* "
                   "(f ลดจากบวกเป็นลบ = เสถียร, ตรงข้าม = ไม่เสถียร, เครื่องหมายเดิม = กึ่งเสถียร)")
    st.markdown("#### สูตรของวิธีเชิงตัวเลข")
    st.latex(r"\text{Euler: } y_{n+1} = y_n + h\,f(x_n, y_n),\quad O(h)")
    st.latex(r"\text{RK4: } y_{n+1} = y_n + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4),\quad O(h^4)")

# ---- 6) การมีอยู่และเอกลักษณ์
with tab_uniq:
    st.markdown("**ทฤษฎีบท (Picard–Lindelöf):** ถ้า f(x, y) และ ∂f/∂y ต่อเนื่องบนสี่เหลี่ยม "
                "R = [x₀−a, x₀+a] × [y₀−b, y₀+b] แล้วปัญหาค่าเริ่มต้น y' = f(x, y), y(x₀) = y₀ "
                "มีคำตอบ **เพียงคำตอบเดียว** บนช่วง |x − x₀| ≤ α โดย α = min(a, b/M), M = max|f| บน R")
    st.latex(r"\frac{\partial f}{\partial y} = " + sp.latex(fy_expr))

    for label, e_ in (("f", expr), ("∂f/∂y", fy_expr)):
        den = oc.denominators(e_)
        if den != 1 and (den.has(oc.X) or den.has(oc.Y)):
            st.warning(f"{label} ไม่นิยามเมื่อ {den} = 0 — ตรวจสอบว่าจุดเริ่มต้นและช่วงที่ใช้ไม่ผ่านบริเวณนี้")

    ra, rb = 0.1 * (x_end - x_start), 0.1 * (ymax - ymin)
    ff, fyf, M, L = oc.grid_check(fn, fy_fn, p, x0 - ra, x0 + ra, y0 - rb, y0 + rb)
    st.markdown(f"**ตรวจรอบจุด (x₀, y₀)** บนสี่เหลี่ยม a = {ra:.4g}, b = {rb:.4g} (สุ่มตัวอย่างตารางจุด)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("f มีค่าจำกัด", f"{ff:.0%}")
    m2.metric("∂f/∂y มีค่าจำกัด", f"{fyf:.0%}")
    m3.metric("M = max|f|", f"{M:.4g}")
    m4.metric("L = max|∂f/∂y|", f"{L:.4g}")
    if ff == 1.0 and fyf == 1.0:
        alpha = ra if M == 0 else min(ra, rb / M)
        st.success(f"เข้าเงื่อนไขของทฤษฎีบท: มีคำตอบเดียวอย่างน้อยบนช่วง |x − x₀| ≤ α ≈ {alpha:.4g} "
                   f"(L ≈ {L:.4g} คือค่าคงตัวลิปชิตซ์ใน y)")
    else:
        st.error("ตรวจพบจุดที่ f หรือ ∂f/∂y ไม่นิยามรอบ (x₀, y₀) — ทฤษฎีบทอาจใช้ไม่ได้ "
                 "จึงอาจมีหลายคำตอบหรือไม่มีคำตอบ")

    ff2, fyf2, _, _ = oc.grid_check(fn, fy_fn, p, x_start, x_end, ymin, ymax)
    st.markdown(f"**ตรวจทั้งหน้าต่างกราฟ:** f มีค่าจำกัด {ff2:.0%}, ∂f/∂y มีค่าจำกัด {fyf2:.0%}")
    st.caption("เป็นการตรวจเชิงตัวเลขจากจุดตัวอย่าง ไม่ใช่การพิสูจน์ "
               "ตัวอย่างที่ทฤษฎีบทใช้ไม่ได้: dy/dx = sqrt(y), y(0) = 0 "
               "(∂f/∂y = 1/(2√y) ไม่นิยามที่ y = 0) มีทั้งคำตอบ y = 0 และ y = x²/4")
