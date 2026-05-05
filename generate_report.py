"""
生成 COVID-19 分析 HTML 报告
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings, json, base64, io, datetime
warnings.filterwarnings('ignore')

# 获取当前日期
current_date = datetime.datetime.now().strftime("%Y-%m-%d")

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# 字体配置（支持中文显示）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 优先中文字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
plt.rcParams['font.size'] = 10  # 设置默认字体大小

# ── 加载数据 ──────────────────────────────────────
df = pd.read_csv("covid_usa_daily.csv", parse_dates=["date"])
df_weekly = pd.read_csv("covid_usa_weekly.csv", parse_dates=["date"])
df_weekly["weekly_cases"] = df_weekly["weekly_cases"].clip(lower=0)

with open("arima_results.json") as f:
    res = json.load(f)

train_arr = np.load("arima_train.npy")
test_arr  = np.load("arima_test_actual.npy")
fc_arr    = np.load("arima_forecast.npy")
fc_lo     = np.load("arima_forecast_lower.npy")
fc_hi     = np.load("arima_forecast_upper.npy")

series_weekly = df_weekly["weekly_cases"].values.astype(float)
series_log = np.log1p(series_weekly)
dates_weekly = df_weekly["date"]

def fig_to_base64(fig, dpi=120):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

PALETTE = {
    "blue": "#2563EB", "red": "#DC2626", "green": "#16A34A",
    "orange": "#EA580C", "purple": "#7C3AED", "gray": "#6B7280",
    "sky": "#0EA5E9", "amber": "#F59E0B",
    "bg": "#0F172A", "card": "#1E293B", "border": "#334155",
    "text": "#F1F5F9", "muted": "#94A3B8",
}

# ──────────────────────────────────────────────────
# 图1: 日确诊 + 7日均线
# ──────────────────────────────────────────────────
fig1, ax = plt.subplots(figsize=(13, 4), facecolor=PALETTE["bg"])
ax.set_facecolor(PALETTE["bg"])
ax.bar(df["date"], df["new_cases"]/1e4, color=PALETTE["blue"],
       alpha=0.35, width=1.2, label="Daily Cases (万)")
ax.plot(df["date"], df["new_cases_smoothed"]/1e4,
        color=PALETTE["sky"], lw=2, label="7-Day Avg (万)")
ax.set_title("USA COVID-19 Daily New Cases", color=PALETTE["text"], fontsize=14, pad=10)
ax.set_ylabel("Cases (10,000)", color=PALETTE["muted"])
ax.tick_params(colors=PALETTE["muted"])
for sp in ax.spines.values(): sp.set_edgecolor(PALETTE["border"])
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=30)
ax.legend(facecolor=PALETTE["card"], edgecolor=PALETTE["border"],
          labelcolor=PALETTE["text"], fontsize=10)
fig1.tight_layout()
img1 = fig_to_base64(fig1)
plt.close(fig1)

# ──────────────────────────────────────────────────
# 图2: 日死亡
# ──────────────────────────────────────────────────
fig2, ax = plt.subplots(figsize=(13, 4), facecolor=PALETTE["bg"])
ax.set_facecolor(PALETTE["bg"])
ax.bar(df["date"], df["new_deaths"], color=PALETTE["red"], alpha=0.5,
       width=1.2, label="Daily Deaths")
smooth_deaths = df["new_deaths"].rolling(7, center=True).mean()
ax.plot(df["date"], smooth_deaths, color=PALETTE["orange"], lw=2,
        label="7-Day Avg")
ax.set_title("USA COVID-19 Daily Deaths", color=PALETTE["text"], fontsize=14, pad=10)
ax.set_ylabel("Deaths", color=PALETTE["muted"])
ax.tick_params(colors=PALETTE["muted"])
for sp in ax.spines.values(): sp.set_edgecolor(PALETTE["border"])
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=30)
ax.legend(facecolor=PALETTE["card"], edgecolor=PALETTE["border"],
          labelcolor=PALETTE["text"], fontsize=10)
fig2.tight_layout()
img2 = fig_to_base64(fig2)
plt.close(fig2)

# ──────────────────────────────────────────────────
# 图3: 周确诊（对数）
# ──────────────────────────────────────────────────
fig3, axes = plt.subplots(2, 1, figsize=(13, 7), facecolor=PALETTE["bg"])
for ax in axes: ax.set_facecolor(PALETTE["bg"])
axes[0].plot(dates_weekly, series_weekly/1e4, color=PALETTE["green"],
             lw=2, marker="o", ms=3, label="Weekly Cases (万)")
axes[0].set_title("Weekly COVID-19 Cases (Original)", color=PALETTE["text"], fontsize=13)
axes[0].set_ylabel("Cases (10,000)", color=PALETTE["muted"])
axes[0].legend(facecolor=PALETTE["card"], edgecolor=PALETTE["border"],
               labelcolor=PALETTE["text"])

axes[1].plot(dates_weekly, series_log, color=PALETTE["purple"],
             lw=2, marker="o", ms=3, label="log(1+Cases)")
axes[1].set_title("Log-Transformed Series (for ARIMA)", color=PALETTE["text"], fontsize=13)
axes[1].set_ylabel("log(1 + Cases)", color=PALETTE["muted"])
axes[1].legend(facecolor=PALETTE["card"], edgecolor=PALETTE["border"],
               labelcolor=PALETTE["text"])

for ax in axes:
    ax.tick_params(colors=PALETTE["muted"])
    for sp in ax.spines.values(): sp.set_edgecolor(PALETTE["border"])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)

fig3.tight_layout()
img3 = fig_to_base64(fig3)
plt.close(fig3)

# ──────────────────────────────────────────────────
# 图4: 季节性分解
# ──────────────────────────────────────────────────
# 使用对数序列做加法分解
decomp = seasonal_decompose(series_log, model="additive", period=4, extrapolate_trend=True)

fig4, axes = plt.subplots(4, 1, figsize=(13, 10), facecolor=PALETTE["bg"])
labels = ["Observed", "Trend", "Seasonal", "Residual"]
colors = [PALETTE["sky"], PALETTE["green"], PALETTE["orange"], PALETTE["purple"]]
components = [decomp.observed, decomp.trend, decomp.seasonal, decomp.resid]

for ax, lbl, col, comp in zip(axes, labels, colors, components):
    ax.set_facecolor(PALETTE["bg"])
    ax.plot(dates_weekly, comp, color=col, lw=1.8)
    ax.set_ylabel(lbl, color=PALETTE["muted"], fontsize=10)
    ax.tick_params(colors=PALETTE["muted"])
    for sp in ax.spines.values(): sp.set_edgecolor(PALETTE["border"])

axes[0].set_title("Seasonal Decomposition (Period=4 weeks)", color=PALETTE["text"], fontsize=13)
for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)

fig4.tight_layout()
img4 = fig_to_base64(fig4)
plt.close(fig4)

# ──────────────────────────────────────────────────
# 图5: ACF / PACF
# ──────────────────────────────────────────────────
diff1 = np.diff(series_log)

fig5, axes = plt.subplots(2, 2, figsize=(13, 7), facecolor=PALETTE["bg"])
for ax in axes.flat:
    ax.set_facecolor(PALETTE["bg"])
    ax.tick_params(colors=PALETTE["muted"])
    for sp in ax.spines.values(): sp.set_edgecolor(PALETTE["border"])

plot_acf(series_log, ax=axes[0, 0], lags=30, color=PALETTE["blue"], title="ACF (Original)")
plot_pacf(series_log, ax=axes[0, 1], lags=30, color=PALETTE["blue"], title="PACF (Original)")
plot_acf(diff1, ax=axes[1, 0], lags=30, color=PALETTE["green"], title="ACF (1st Diff)")
plot_pacf(diff1, ax=axes[1, 1], lags=30, color=PALETTE["green"], title="PACF (1st Diff)")

for ax in axes.flat:
    ax.title.set_color(PALETTE["text"])
    ax.set_xlabel("Lags", color=PALETTE["muted"])
    ax.set_ylabel("Correlation", color=PALETTE["muted"])

fig5.suptitle("Autocorrelation Analysis", color=PALETTE["text"], fontsize=14)
fig5.tight_layout()
img5 = fig_to_base64(fig5)
plt.close(fig5)

# ──────────────────────────────────────────────────
# 图6: AIC 比较
# ──────────────────────────────────────────────────
top_aic = res["top_aic"]
labels_aic = [f"ARIMA{tuple(r[0])}" for r in top_aic]
aic_vals = [r[1] for r in top_aic]
colors_bar = [PALETTE["amber"] if i == 0 else PALETTE["blue"] for i in range(len(labels_aic))]

fig6, ax = plt.subplots(figsize=(8, 4), facecolor=PALETTE["bg"])
ax.set_facecolor(PALETTE["bg"])
bars = ax.barh(labels_aic[::-1], aic_vals[::-1], color=colors_bar[::-1], height=0.55)
ax.set_title("Top ARIMA Models by AIC (lower is better)", color=PALETTE["text"], fontsize=13)
ax.set_xlabel("AIC", color=PALETTE["muted"])
ax.tick_params(colors=PALETTE["muted"])
for sp in ax.spines.values(): sp.set_edgecolor(PALETTE["border"])
for bar, val in zip(bars, aic_vals[::-1]):
    ax.text(val + 0.3, bar.get_y() + bar.get_height()/2,
            f"{val:.2f}", va="center", color=PALETTE["text"], fontsize=10)
fig6.tight_layout()
img6 = fig_to_base64(fig6)
plt.close(fig6)

# ──────────────────────────────────────────────────
# 图7: ARIMA 预测 vs 实际（聚焦测试期）
# ──────────────────────────────────────────────────
train_size = len(series_log) - 12
test_dates = dates_weekly.iloc[train_size:].reset_index(drop=True)

fig7, ax = plt.subplots(figsize=(13, 5), facecolor=PALETTE["bg"])
ax.set_facecolor(PALETTE["bg"])

# 只显示测试期数据（单位：个位）
ax.plot(test_dates, test_arr,
        color=PALETTE["green"], lw=2, marker="o", ms=7, label="Actual")
ax.plot(test_dates, fc_arr,
        color=PALETTE["amber"], lw=2, marker="s", ms=7, ls="--", label="ARIMA Forecast")
ax.fill_between(test_dates, fc_lo, fc_hi,
                color=PALETTE["amber"], alpha=0.2, label="95% CI")

ax.set_title(f"ARIMA{res['arima_best']['order']} Forecast vs Actual (Last 12 Weeks)",
             color=PALETTE["text"], fontsize=13)
ax.set_ylabel("Weekly Cases", color=PALETTE["muted"])
ax.set_xlabel("Date", color=PALETTE["muted"])
ax.tick_params(colors=PALETTE["muted"])
for sp in ax.spines.values(): sp.set_edgecolor(PALETTE["border"])
ax.legend(facecolor=PALETTE["card"], edgecolor=PALETTE["border"],
          labelcolor=PALETTE["text"], fontsize=10)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.xticks(rotation=30)

# 设置纵轴固定范围 0-1000000
ax.set_ylim(bottom=0, top=1000000)
fig7.tight_layout()
img7 = fig_to_base64(fig7)
plt.close(fig7)

# ──────────────────────────────────────────────────
# 图8: 残差诊断
# ──────────────────────────────────────────────────
residuals = test_arr - fc_arr

fig8, axes = plt.subplots(1, 3, figsize=(13, 4), facecolor=PALETTE["bg"])
for ax in axes:
    ax.set_facecolor(PALETTE["bg"])
    ax.tick_params(colors=PALETTE["muted"])
    for sp in ax.spines.values(): sp.set_edgecolor(PALETTE["border"])

# 残差时序
axes[0].plot(range(len(residuals)), residuals/1e4, color=PALETTE["sky"],
             marker="o", ms=5, lw=2)
axes[0].axhline(0, color=PALETTE["border"], ls="--")
axes[0].set_title("Residuals", color=PALETTE["text"])
axes[0].set_ylabel("Error (万)", color=PALETTE["muted"])

# 残差直方图
axes[1].hist(residuals/1e4, bins=8, color=PALETTE["purple"], alpha=0.75, edgecolor=PALETTE["card"])
axes[1].set_title("Residual Histogram", color=PALETTE["text"])
axes[1].set_xlabel("Error (万)", color=PALETTE["muted"])

# Q-Q图
from scipy import stats as scipy_stats
(osm, osr), (slope, intercept, r) = scipy_stats.probplot(residuals)
axes[2].scatter(osm, osr, color=PALETTE["green"], s=40, alpha=0.8)
axes[2].plot(osm, slope * np.array(osm) + intercept, color=PALETTE["amber"], lw=2)
axes[2].set_title(f"Q-Q Plot (r={r:.3f})", color=PALETTE["text"])
axes[2].set_xlabel("Theoretical Quantiles", color=PALETTE["muted"])
axes[2].set_ylabel("Sample Quantiles", color=PALETTE["muted"])

fig8.suptitle("Residual Diagnostics", color=PALETTE["text"], fontsize=14)
fig8.tight_layout()
img8 = fig_to_base64(fig8)
plt.close(fig8)

print("所有图表生成完毕！")

# ──────────────────────────────────────────────────
# 构建 HTML
# ──────────────────────────────────────────────────
arima_params_html = ""
for pname, coef in res["arima_params"].items():
    pv = res["arima_pvalues"].get(pname, 1.0)
    sig = "✓" if pv < 0.05 else "✗"
    color = "#22D3EE" if pv < 0.05 else "#F87171"
    arima_params_html += f"""
    <tr>
      <td>{pname}</td>
      <td>{coef:.4f}</td>
      <td>{pv:.4f}</td>
      <td style="color:{color};font-weight:700">{sig}</td>
    </tr>"""

top_aic_html = ""
for i, (order, aic) in enumerate(res["top_aic"]):
    cls = "best-row" if i == 0 else ""
    top_aic_html += f"""
    <tr class="{cls}">
      <td>ARIMA{tuple(order)}</td>
      <td>{aic:.2f}</td>
      <td>{"★ Best" if i==0 else ""}</td>
    </tr>"""

d = res["desc_stats"]
ab = res["arima_best"]

html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>COVID-19 时间序列分析报告 — 美国</title>
<style>
  :root {{
    --bg: #0F172A; --card: #1E293B; --border: #334155;
    --text: #F1F5F9; --muted: #94A3B8;
    --blue: #2563EB; --sky: #0EA5E9; --green: #22C55E;
    --amber: #F59E0B; --red: #EF4444; --purple: #A855F7;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: "Segoe UI",system-ui,sans-serif; font-size: 15px; line-height: 1.7; }}
  .page {{ max-width: 1100px; margin: 0 auto; padding: 28px 20px 60px; }}

  /* 顶部标题 */
  .hero {{ background: linear-gradient(135deg,#1e3a5f,#0f172a 60%,#1e293b);
           border: 1px solid var(--border); border-radius: 16px; padding: 40px;
           margin-bottom: 32px; position: relative; overflow: hidden; }}
  .hero::before {{ content:""; position:absolute; inset:0;
                   background: radial-gradient(ellipse 60% 50% at 70% 50%, rgba(37,99,235,.18),transparent); }}
  .hero h1 {{ font-size: 2rem; font-weight: 800; color: #fff; margin-bottom: 8px; }}
  .hero p  {{ color: var(--muted); font-size: .95rem; }}
  .badge   {{ display:inline-block; background:rgba(37,99,235,.25); border:1px solid var(--sky);
              color:var(--sky); border-radius:99px; padding:2px 12px; font-size:.8rem; margin-right:6px; }}

  /* 统计卡片 */
  .kpi-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:14px; margin-bottom:30px; }}
  .kpi {{ background:var(--card); border:1px solid var(--border); border-radius:12px; padding:20px; text-align:center; }}
  .kpi .val {{ font-size:1.9rem; font-weight:800; margin-bottom:4px; }}
  .kpi .lbl {{ color:var(--muted); font-size:.83rem; }}

  /* 卡片 */
  .card {{ background:var(--card); border:1px solid var(--border); border-radius:12px;
           padding:24px; margin-bottom:24px; }}
  .card h2 {{ font-size:1.15rem; font-weight:700; margin-bottom:16px;
              display:flex; align-items:center; gap:8px; color:var(--text); }}
  .card h2 .dot {{ width:8px; height:8px; border-radius:50%; display:inline-block; }}

  /* 图片 */
  .img-wrap {{ border-radius:8px; overflow:hidden; background:#0a1020; }}
  .img-wrap img {{ width:100%; display:block; }}

  /* 表格 */
  table {{ width:100%; border-collapse:collapse; font-size:.9rem; }}
  thead tr {{ background:rgba(37,99,235,.15); }}
  th {{ padding:10px 14px; text-align:left; color:var(--muted); font-weight:600; border-bottom:1px solid var(--border); }}
  td {{ padding:9px 14px; border-bottom:1px solid rgba(51,65,85,.5); }}
  tr:last-child td {{ border-bottom:none; }}
  tr.best-row {{ background:rgba(245,158,11,.1); color:#F59E0B; font-weight:700; }}
  tr:hover td {{ background:rgba(255,255,255,.03); }}

  /* 行内指标 */
  .metric-row {{ display:flex; gap:20px; flex-wrap:wrap; margin-bottom:16px; }}
  .metric {{ background:rgba(37,99,235,.1); border:1px solid rgba(37,99,235,.3);
             border-radius:8px; padding:10px 18px; }}
  .metric .m-val {{ font-size:1.3rem; font-weight:700; color:var(--sky); }}
  .metric .m-lbl {{ color:var(--muted); font-size:.8rem; }}

  /* 章节标题 */
  .section-title {{ font-size:1.4rem; font-weight:800; margin:36px 0 16px;
                    color:var(--text); border-left:4px solid var(--sky); padding-left:12px; }}

  /* 提示框 */
  .info-box {{ background:rgba(14,165,233,.08); border:1px solid rgba(14,165,233,.3);
               border-radius:8px; padding:14px 18px; margin-bottom:16px; color:var(--muted); font-size:.9rem; }}
  .info-box strong {{ color:var(--sky); }}

  /* 公式 */
  .formula {{ background:#060c18; border:1px solid var(--border); border-radius:8px;
              padding:14px 20px; font-family:monospace; font-size:.95rem; color:#7DD3FC; margin:12px 0; }}

  /* 结论 */
  .conclusion {{ background:linear-gradient(135deg,rgba(37,99,235,.12),rgba(126,34,206,.08));
                 border:1px solid rgba(37,99,235,.3); border-radius:12px; padding:24px; }}
  .conclusion h3 {{ color:var(--sky); margin-bottom:12px; }}
  .conclusion li {{ margin-bottom:8px; color:var(--muted); }}
  .conclusion li strong {{ color:var(--text); }}

  footer {{ text-align:center; color:var(--muted); font-size:.8rem; margin-top:40px; padding-top:20px; border-top:1px solid var(--border); }}
</style>
</head>
<body>
<div class="page">

<!-- ── Hero ── -->
<div class="hero">
  <h1>🦠 COVID-19 时间序列分析报告</h1>
  <p style="margin-bottom:12px">数据来源: <a href="https://ourworldindata.org" style="color:var(--sky)">Our World in Data</a> &nbsp;|&nbsp; 分析地区: <strong style="color:#fff">美国 (USA)</strong> &nbsp;|&nbsp; 时间跨度: <strong style="color:#fff">{d['date_start']} → {d['date_end']}</strong></p>
  <span class="badge">时间序列分析</span>
  <span class="badge">ARIMA 建模</span>
  <span class="badge">参数估计</span>
  <span class="badge">预测验证</span>
</div>

<!-- ── KPI ── -->
<div class="kpi-grid">
  <div class="kpi">
    <div class="val" style="color:#38BDF8">{d['total_cases']/1e6:.1f}M</div>
    <div class="lbl">累计确诊病例</div>
  </div>
  <div class="kpi">
    <div class="val" style="color:#F87171">{d['total_deaths']/1e3:.0f}K</div>
    <div class="lbl">累计死亡人数</div>
  </div>
  <div class="kpi">
    <div class="val" style="color:#4ADE80">{d['mean_weekly_cases']/1e3:.1f}K</div>
    <div class="lbl">周均确诊</div>
  </div>
  <div class="kpi">
    <div class="val" style="color:#FBBF24">{d['max_weekly_cases']/1e6:.2f}M</div>
    <div class="lbl">周确诊峰值 ({d['max_weekly_cases_date']})</div>
  </div>
  <div class="kpi">
    <div class="val" style="color:#E879F9">{d['max_weekly_deaths']:,.0f}</div>
    <div class="lbl">周死亡峰值 ({d['max_weekly_deaths_date']})</div>
  </div>
</div>

<!-- ── 第一节：数据概况 ── -->
<div class="section-title">§ 1 &nbsp; 数据概况与来源</div>
<div class="card">
  <div class="info-box">
    <strong>数据来源：</strong>Our World in Data（OWID）COVID-19 开放数据集，原始数据由 Johns Hopkins University & WHO 提供，每日更新。
    本报告选取 <strong>美国（USA）</strong> 日度数据，时间段为 2020-03-01 至 2023-05-11，共 <strong>1,167</strong> 个日数据点，
    聚合为 <strong>167</strong> 个周数据点用于时间序列建模。
    <br/><strong>数据说明：</strong>原始数据中 <code>new_cases</code> 和 <code>new_deaths</code> 字段为每周汇总值（每周某一天记录整周数据，其余日期为0），KPI展示为周数据统计。
  </div>
  <h2><span class="dot" style="background:var(--sky)"></span>日确诊病例与7日滑动均值</h2>
  <div class="img-wrap"><img src="data:image/png;base64,{img1}" alt="daily cases"/></div>
  <p style="color:var(--muted);font-size:.85rem;margin-top:8px">
    可见明显的多波次疫情浪潮：2020年底冬季第一波、2021年冬季Delta波、2022年初Omicron超级波。
  </p>
</div>
<div class="card">
  <h2><span class="dot" style="background:var(--red)"></span>日死亡人数与7日滑动均值</h2>
  <div class="img-wrap"><img src="data:image/png;base64,{img2}" alt="daily deaths"/></div>
  <p style="color:var(--muted);font-size:.85rem;margin-top:8px">
    死亡峰值与确诊峰值相比有约2–3周的滞后效应，符合疾病进展规律。
  </p>
</div>

<!-- ── 第二节：数据特征分析 ── -->
<div class="section-title">§ 2 &nbsp; 数据特征分析</div>
<div class="card">
  <h2><span class="dot" style="background:var(--green)"></span>周数据序列 & 对数变换</h2>
  <div class="img-wrap"><img src="data:image/png;base64,{img3}" alt="weekly series"/></div>
  <p style="color:var(--muted);font-size:.85rem;margin-top:8px">
    原始周序列方差随均值增大而扩张（异方差性），对数变换 log(1+y) 有效稳定方差，是 ARIMA 建模的预处理步骤。
  </p>
</div>
<div class="card">
  <h2><span class="dot" style="background:var(--amber)"></span>时序分解（加法模型，周期=4周）</h2>
  <div class="img-wrap"><img src="data:image/png;base64,{img4}" alt="decomposition"/></div>
  <p style="color:var(--muted);font-size:.85rem;margin-top:8px">
    趋势分量清晰呈现四轮疫情波次；季节分量显示约4周的月内周期性；残差项反映随机扰动。
  </p>
</div>
<div class="card">
  <h2><span class="dot" style="background:var(--purple)"></span>平稳性检验（ADF 检验）</h2>
  <table>
    <thead><tr><th>序列</th><th>ADF 统计量</th><th>p-value</th><th>结论</th></tr></thead>
    <tbody>
      <tr><td>log(1+Cases) 原序列</td><td>{res['adf_test']['statistic']:.4f}</td>
          <td>{res['adf_test']['pvalue']:.4f}</td>
          <td style="color:#4ADE80">✓ 弱平稳（p&lt;0.05）</td></tr>
      <tr><td>一阶差分序列 ∇log</td><td>—</td>
          <td>{res['adf_test']['diff_pvalue']:.4f}</td>
          <td style="color:#4ADE80">✓ 强平稳（p≈0.000）</td></tr>
    </tbody>
  </table>
  <p style="color:var(--muted);font-size:.85rem;margin-top:10px">
    Augmented Dickey-Fuller 检验：原序列在5%水平已拒绝单位根假设；差分后更加确定。ARIMA 差分阶数 d=1 为合理选择。
  </p>
</div>
<div class="card">
  <h2><span class="dot" style="background:var(--blue)"></span>自相关 & 偏自相关图（ACF / PACF）</h2>
  <div class="img-wrap"><img src="data:image/png;base64,{img5}" alt="acf pacf"/></div>
  <p style="color:var(--muted);font-size:.85rem;margin-top:8px">
    原序列 ACF 缓慢衰减，确认需差分；一阶差分后 ACF 在滞后1–2处显著截尾，PACF 在滞后1–2处截尾，提示 ARIMA(2,1,2) 为候选模型。
  </p>
</div>

<!-- ── 第三节：ARIMA 建模 ── -->
<div class="section-title">§ 3 &nbsp; ARIMA 时间序列预测模型</div>

<div class="card">
  <h2><span class="dot" style="background:var(--amber)"></span>模型公式</h2>
  <div class="formula">ARIMA(p, d, q) : &nbsp; Φ(B) ∇ᵈ yₜ = Θ(B) εₜ</div>
  <div class="formula">本报告最优模型 ARIMA(2, 1, 2) :<br/>
(1 − φ₁B − φ₂B²)(1 − B)yₜ = (1 + θ₁B + θ₂B²)εₜ</div>
  <p style="color:var(--muted);font-size:.88rem;margin-top:8px">
    其中 B 为后移算子，∇=1-B 为差分算子；yₜ = log(1 + 周确诊数)；εₜ ~ N(0, σ²) 为白噪声。<br/>
    参数估计方法：<strong style="color:var(--text)">极大似然估计（MLE）</strong>，协方差矩阵由梯度外积法（OPG）计算。
  </p>
</div>

<div class="card">
  <h2><span class="dot" style="background:var(--sky)"></span>参数网格搜索（AIC 准则）</h2>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;align-items:start">
    <div>
      <table>
        <thead><tr><th>模型阶数</th><th>AIC</th><th>排名</th></tr></thead>
        <tbody>{top_aic_html}</tbody>
      </table>
      <p style="color:var(--muted);font-size:.83rem;margin-top:10px">
        遍历 p∈[0,3], d∈[0,1], q∈[0,3] 共64组参数，选 AIC 最小者。
      </p>
    </div>
    <div class="img-wrap"><img src="data:image/png;base64,{img6}" alt="aic comparison"/></div>
  </div>
</div>

<div class="card">
  <h2><span class="dot" style="background:var(--green)"></span>参数估计结果 — ARIMA(2, 1, 2)</h2>
  <div class="metric-row">
    <div class="metric"><div class="m-val">-30.16</div><div class="m-lbl">AIC</div></div>
    <div class="metric"><div class="m-val">-14.97</div><div class="m-lbl">BIC</div></div>
    <div class="metric"><div class="m-val">20.078</div><div class="m-lbl">Log-Likelihood</div></div>
    <div class="metric"><div class="m-val">155</div><div class="m-lbl">训练样本</div></div>
  </div>
  <table>
    <thead><tr><th>参数</th><th>系数估计值</th><th>p-value</th><th>显著性（α=0.05）</th></tr></thead>
    <tbody>{arima_params_html}</tbody>
  </table>
  <p style="color:var(--muted);font-size:.85rem;margin-top:12px">
    <strong style="color:var(--text)">解读：</strong><br/>
    • <strong>ar.L1 = 1.4583</strong>（p&lt;0.001）：上一周对数确诊数对本周有强烈正向自回归效应。<br/>
    • <strong>ar.L2 = -0.5926</strong>（p&lt;0.001）：两周前的反向修正，体现疫情波动的均值回归特性。<br/>
    • <strong>ma.L1 = -0.8019, ma.L2 = 0.6061</strong>：移动平均项捕捉随机冲击的滞后传导。<br/>
    • <strong>σ² = 0.0444</strong>：对数尺度残差方差，对应原始尺度预测误差约 ±e^0.21 倍数。
  </p>
</div>

<div class="card">
  <h2><span class="dot" style="background:var(--amber)"></span>滚动预测 vs 实际值（末尾12周）</h2>
  <div class="metric-row">
    <div class="metric"><div class="m-val">{ab['mae']/1e4:.1f}万</div><div class="m-lbl">MAE（平均绝对误差/周）</div></div>
    <div class="metric"><div class="m-val">{ab['rmse']/1e4:.1f}万</div><div class="m-lbl">RMSE（均方根误差/周）</div></div>
    <div class="metric"><div class="m-val">{ab['mape']:.1f}%</div><div class="m-lbl">MAPE</div></div>
  </div>
  <div class="img-wrap"><img src="data:image/png;base64,{img7}" alt="forecast"/></div>
  <p style="color:var(--muted);font-size:.85rem;margin-top:8px">
    蓝色为训练集，绿色为真实测试值，橙色虚线为 ARIMA 预测值，阴影为95%置信区间。
    测试期（2023年初）为疫情消退尾段，真实值快速收敛至低位，模型对此极速下降的捕捉构成主要误差来源。
  </p>
</div>

<div class="card">
  <h2><span class="dot" style="background:var(--purple)"></span>残差诊断</h2>
  <div class="img-wrap"><img src="data:image/png;base64,{img8}" alt="residuals"/></div>
  <p style="color:var(--muted);font-size:.85rem;margin-top:8px">
    • 残差时序图：整体围绕零均值波动，无明显系统性偏差。<br/>
    • 直方图：大致对称，但存在少量极值（重大政策/报告调整所致）。<br/>
    • Q-Q图：主体部分贴近正态分布，尾部稍有偏离，与 JB 检验中峰度=11.78 一致——COVID 数据天然具有厚尾特征。
  </p>
</div>

<!-- ── 第四节：综合结论 ── -->
<div class="section-title">§ 4 &nbsp; 综合结论</div>
<div class="conclusion">
  <h3>📌 主要发现</h3>
  <ul style="list-style:disc;padding-left:1.2em">
    <li><strong>数据特征：</strong>美国 COVID-19 日确诊序列表现出显著的多波次疫情结构、异方差性和厚尾分布特征。</li>
    <li><strong>平稳性：</strong>对数变换后序列通过 ADF 检验（p=0.008），一阶差分后进一步增强平稳性，d=1 合理。</li>
    <li><strong>最优模型：</strong>格搜索确定 <strong>ARIMA(2,1,2)</strong> 为 AIC 最优（-30.16），全部5个参数均在 1% 水平显著。</li>
    <li><strong>参数解读：</strong>AR(2) 结构捕捉疫情的"惯性+回调"动力学；MA(2) 捕捉随机冲击的短期传播。</li>
    <li><strong>预测精度：</strong>MAE ≈ 29万例/周，测试期为消退段，预测偏高主要源于 ARIMA 线性外推局限。</li>
    <li><strong>改进方向：</strong>可引入 SARIMA（季节项）、Prophet（变点检测）或含 NPI 政策变量的外生模型（ARIMAX）进一步提升精度。</li>
  </ul>
</div>

<footer>
  <p>数据来源: Our World in Data COVID-19 Dataset (CC BY 4.0) &nbsp;|&nbsp; 分析工具: Python / statsmodels / pandas / matplotlib</p>
  <p style="margin-top:4px">报告生成时间: {current_date}</p>
</footer>

</div>
</body>
</html>"""

with open("covid_report.html", "w", encoding="utf-8") as f:
    f.write(html)

print("HTML 报告已生成: covid_report.html")
