"""
COVID-19 时间序列分析与预测模型
数据来源: Our World in Data (OWID)
分析地区: 美国 (USA)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rcParams
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
import io
import base64
import json

# ────────────────────────────────────────────────
# 1. 数据获取
# ────────────────────────────────────────────────
print("=" * 60)
print("正在从 Our World in Data 获取 COVID-19 数据...")
print("=" * 60)

url = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"

import urllib.request

req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
response = urllib.request.urlopen(req, timeout=60)
raw_content = response.read()
print(f"数据下载完成，大小: {len(raw_content)/1024/1024:.1f} MB")

df_all = pd.read_csv(io.BytesIO(raw_content), low_memory=False)
print(f"总记录数: {len(df_all):,}")
print(f"国家/地区数: {df_all['location'].nunique()}")

# ────────────────────────────────────────────────
# 2. 筛选美国数据
# ────────────────────────────────────────────────
df = df_all[df_all["iso_code"] == "USA"].copy()
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# 选取核心字段
cols = ["date", "new_cases", "new_deaths", "new_cases_smoothed",
        "new_deaths_smoothed", "total_cases", "total_deaths",
        "hosp_patients", "reproduction_rate"]
df = df[cols]

# 截取有效时段 (2020-03-01 至 2023-05-11)
df = df[(df["date"] >= "2020-03-01") & (df["date"] <= "2023-05-11")].reset_index(drop=True)

# 填补少量缺失
df["new_cases"] = df["new_cases"].fillna(0).clip(lower=0)
df["new_deaths"] = df["new_deaths"].fillna(0).clip(lower=0)
df["new_cases_smoothed"] = df["new_cases_smoothed"].fillna(method="ffill").fillna(0)

# 生成周聚合数据
df_weekly = df.resample("W", on="date").agg(
    weekly_cases=("new_cases", "sum"),
    weekly_deaths=("new_deaths", "sum"),
    avg_reproduction=("reproduction_rate", "mean")
).reset_index()
df_weekly = df_weekly[df_weekly["weekly_cases"] > 0].reset_index(drop=True)

print(f"\n美国数据时段: {df['date'].min().date()} → {df['date'].max().date()}")
print(f"日数据点数: {len(df)}")
print(f"周数据点数: {len(df_weekly)}")
print("\n前5行:")
print(df.head())

# ────────────────────────────────────────────────
# 3. 描述性统计
# ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("描述性统计")
print("=" * 60)
desc = df[["new_cases", "new_deaths"]].describe()
print(desc)

# 峰值
peak_cases_idx = df["new_cases"].idxmax()
peak_deaths_idx = df["new_deaths"].idxmax()
print(f"\n单日确诊峰值: {df.loc[peak_cases_idx,'new_cases']:,.0f} ({df.loc[peak_cases_idx,'date'].date()})")
print(f"单日死亡峰值: {df.loc[peak_deaths_idx,'new_deaths']:,.0f} ({df.loc[peak_deaths_idx,'date'].date()})")

# ────────────────────────────────────────────────
# 4. 使用周数据建立 ARIMA 模型
# ────────────────────────────────────────────────
# 使用对数变换平稳化
series = df_weekly["weekly_cases"].values.astype(float)
series_log = np.log1p(series)

# 平稳性检验
print("\n" + "=" * 60)
print("平稳性检验 (ADF Test)")
print("=" * 60)
adf_result = adfuller(series_log, autolag="AIC")
print(f"ADF Statistic: {adf_result[0]:.4f}")
print(f"p-value: {adf_result[1]:.4f}")
print(f"Critical Values: {adf_result[4]}")
if adf_result[1] < 0.05:
    print("结论: 序列平稳（拒绝单位根假设）")
else:
    print("结论: 序列非平稳，需要差分")

# 一阶差分
series_diff = np.diff(series_log)
adf_diff = adfuller(series_diff, autolag="AIC")
print(f"\n一阶差分后 ADF p-value: {adf_diff[1]:.4f}")
if adf_diff[1] < 0.05:
    print("一阶差分后序列平稳")

# ────────────────────────────────────────────────
# 5. ARIMA 参数选择 (基于 AIC)
# ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ARIMA 参数网格搜索 (p,d,q)")
print("=" * 60)

best_aic = np.inf
best_order = None
results_grid = []

for p in range(0, 4):
    for d in range(0, 2):
        for q in range(0, 4):
            try:
                model = ARIMA(series_log, order=(p, d, q))
                fitted = model.fit()
                aic = fitted.aic
                results_grid.append((p, d, q, aic))
                if aic < best_aic:
                    best_aic = aic
                    best_order = (p, d, q)
            except:
                pass

results_grid.sort(key=lambda x: x[3])
print("Top 5 参数组合 (by AIC):")
print(f"{'Order':<15} {'AIC':>10}")
print("-" * 26)
for r in results_grid[:5]:
    print(f"ARIMA{r[:3]}  {r[3]:>10.2f}")
print(f"\n最优参数: ARIMA{best_order}, AIC = {best_aic:.2f}")

# ────────────────────────────────────────────────
# 6. 拟合最优 ARIMA 模型
# ────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"拟合 ARIMA{best_order} 模型")
print("=" * 60)

# 训练/测试分割 (最后 12 周作为测试集)
train_size = len(series_log) - 12
train = series_log[:train_size]
test = series_log[train_size:]

arima_model = ARIMA(train, order=best_order)
arima_fitted = arima_model.fit()

print(arima_fitted.summary())

# 预测
forecast_result = arima_fitted.get_forecast(steps=12)
forecast_mean_log = forecast_result.predicted_mean
forecast_ci_log = forecast_result.conf_int(alpha=0.05)

# 反变换
forecast_mean = np.expm1(forecast_mean_log)
# 处理置信区间（可能是 DataFrame 或 ndarray）
if hasattr(forecast_ci_log, 'iloc'):
    forecast_lower = np.expm1(forecast_ci_log.iloc[:, 0])
    forecast_upper = np.expm1(forecast_ci_log.iloc[:, 1])
else:
    forecast_lower = np.expm1(forecast_ci_log[:, 0])
    forecast_upper = np.expm1(forecast_ci_log[:, 1])
test_actual = np.expm1(test)

# 评估指标
mae = mean_absolute_error(test_actual, forecast_mean)
rmse = np.sqrt(mean_squared_error(test_actual, forecast_mean))
forecast_mean_arr = np.array(forecast_mean).flatten()
mape = np.mean(np.abs((test_actual - forecast_mean_arr) / (test_actual + 1))) * 100

print(f"\n预测评估 (最后12周):")
print(f"  MAE:  {mae:,.0f} 例/周")
print(f"  RMSE: {rmse:,.0f} 例/周")
print(f"  MAPE: {mape:.2f}%")

# ────────────────────────────────────────────────
# 7. 保存数据用于报告
# ────────────────────────────────────────────────
results = {
    "desc_stats": {
        "mean_daily_cases": float(df["new_cases"].mean()),
        "max_daily_cases": float(df["new_cases"].max()),
        "max_daily_cases_date": str(df.loc[peak_cases_idx, "date"].date()),
        "mean_daily_deaths": float(df["new_deaths"].mean()),
        "max_daily_deaths": float(df["new_deaths"].max()),
        "max_daily_deaths_date": str(df.loc[peak_deaths_idx, "date"].date()),
        "total_cases": float(df["total_cases"].max()),
        "total_deaths": float(df["total_deaths"].max()),
        "date_start": str(df["date"].min().date()),
        "date_end": str(df["date"].max().date()),
    },
    "adf_test": {
        "statistic": float(adf_result[0]),
        "pvalue": float(adf_result[1]),
        "diff_pvalue": float(adf_diff[1]),
    },
    "arima_best": {
        "order": best_order,
        "aic": float(best_aic),
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
    },
    "arima_params": {
        name: float(val) for name, val in zip(
            arima_fitted.param_names, arima_fitted.params
        )
    },
    "arima_pvalues": {
        name: float(val) for name, val in zip(
            arima_fitted.param_names, arima_fitted.pvalues
        )
    },
    "top_aic": [(list(r[:3]), float(r[3])) for r in results_grid[:5]],
}

# 保存关键数组
np.save("arima_train.npy", np.expm1(train))
np.save("arima_test_actual.npy", np.array(test_actual))
np.save("arima_forecast.npy", np.array(forecast_mean).flatten())
np.save("arima_forecast_lower.npy", np.array(forecast_lower).flatten())
np.save("arima_forecast_upper.npy", np.array(forecast_upper).flatten())

df.to_csv("covid_usa_daily.csv", index=False)
df_weekly.to_csv("covid_usa_weekly.csv", index=False)

with open("arima_results.json", "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\n数据和结果已保存。")
print("=" * 60)
print("分析完成！")
