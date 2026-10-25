# LLM 行为与心理测量研究代码

从文本检测项目复制的研究基础代码：固定原始／中性／人类假设／机器假设四种前缀，测量对齐正文 Token 的概率变化；包含 Token 置信度统计、条件差分和 CPU 基线分析。尚不是完整的人格或心理量表测评系统。

## 目录

- `src/b_text_detector/`：原始研究模块及依赖，保留包名以兼容脚本。
- `scripts/`：条件测量、置信度提取和基线实验入口。
- `configs/`：原实验配置。
- `tests/`：对应单元测试。
- `reports/`：原实验协议、特征说明和结果报告。
- `../参考资料/`：仓库根目录原有研究思路，保持不变。
- `COPY_PROVENANCE.json`：来源路径及逐文件 SHA-256。

## 运行

先进入 `ai文章检测/` 目录，再准备匹配 CUDA 的 PyTorch、安装依赖：

```bash
cd ~/llm-behavior-psychometrics/ai文章检测
python -m pip install -r requirements.txt
python -m pytest -q
PYTHONPATH=src python scripts/run_condition_probe.py --help
PYTHONPATH=src python scripts/run_condition_baseline.py --help
```

当前是原代码复制，未重新训练或提取全量特征。没有复制模型权重、真实数据和特征缓存。实验配置中的 `artifacts/...` 为原项目布局；正式复现实验前须准备对应数据、切分、特征与协议产物，并核对配置和脚本路径。部分历史启动／守护脚本保留原环境路径，应检查后再执行。测量引擎与测试可独立导入，无需原项目源码。

源代码位置：`/home/revy/数据挖掘/b_text_detector/research/legacy/`。源文件保留；本次没有提交或推送 Git。
