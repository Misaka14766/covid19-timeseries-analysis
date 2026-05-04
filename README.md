# COVID-19 时间序列分析与预测模型

> 数据来源：[Our World in Data](https://ourworldindata.org/covid-cases) | 分析地区：**美国（USA）** | 时间跨度：2020-03-01 → 2023-05-11

## 项目简介

本项目从互联网获取 COVID-19 真实数据，进行时间序列特征分析，建立 **ARIMA(2,1,2)** 预测模型并完成参数估计，最终输出完整的可视化分析报告。

## 分析内容

| 模块 | 内容 |
|------|------|
| 数据获取 | OWID 开放数据集，1,167 日 / 167 周数据点 |
| 描述性统计 | 均值、峰值、多波次疫情结构分析 |
| 数据变换 | log(1+y) 对数变换稳定异方差 |
| 平稳性检验 | ADF 检验（p=0.008，弱平稳） |
| 季节性分解 | 趋势 + 季节 + 残差分量 |
| ACF/PACF 分析 | 自相关图辅助定阶 |
| 模型选择 | 网格搜索 64 组参数，AIC 准则 |
| 最优模型 | **ARIMA(2,1,2)**，AIC = -30.16 |
| 参数估计 | MLE 极大似然估计，全部参数 p<0.001 |
| 预测评估 | MAE ≈ 29.3 万例/周（末尾 12 周测试集） |

## ARIMA(2,1,2) 参数估计结果

```
ar.L1  =  1.4583  (p < 0.001)
ar.L2  = -0.5926  (p < 0.001)
ma.L1  = -0.8019  (p < 0.001)
ma.L2  =  0.6061  (p < 0.001)
σ²     =  0.0444  (p < 0.001)

AIC = -30.16  |  BIC = -14.97  |  Log-Likelihood = 20.08
```

## 文件说明

```
covid_analysis.py       # 数据获取 + ARIMA 建模脚本
generate_report.py      # HTML 报告生成脚本
covid_report.html       # 完整可视化分析报告（含8张图表）
covid_usa_daily.csv     # 美国日度数据
covid_usa_weekly.csv    # 美国周度聚合数据
arima_results.json      # 模型参数与评估指标
*.npy                   # 训练/测试/预测数组
```

## 快速运行

```bash
pip install pandas numpy matplotlib scipy statsmodels scikit-learn seaborn
python covid_analysis.py    # 下载数据并训练模型
python generate_report.py   # 生成 HTML 报告
```

## 技术栈

- **Python 3.12**
- **statsmodels** — ARIMA 建模与参数估计
- **pandas / numpy** — 数据处理
- **matplotlib** — 可视化
- **scipy** — 统计检验

## 数据许可

原始数据来自 Our World in Data，遵循 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 协议。
