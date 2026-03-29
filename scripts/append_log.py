import os

log_entry = """
### 2026-03-26: 第五轮报告修正（1D表面变量与扰动场重审）
**修改原因**: 在前四轮全面检查后，复查发现 1D 表面通量图（尤其是 OLR 和柱积分扰动场）的 FDR 值由于网格规模差异（144点 vs 3456点）在前四轮替换中被遗漏。此外，FDR 的大幅变化导致原文相关的定性结论失效。
**修改文件**: 
- `C:\\Users\\Lenovo\\Desktop\\诊断图总结报告_修正版.docx`
**修改内容**: 
- 更新了 8 处遗漏的 FDR 数值：`bg_corr_olr` (13.3%→39.6%), `bg_diff_olr` (11.6%→26.4%), `mjo_corr_olr` (3.2%→4.8%), `mjo_diff_olr` (0.8%→2.1%), `mjo_corr_column_q` (8.3%→32.4%), `mjo_diff_column_q` (7.5%→26.2%), `mjo_corr_column_mse` (5.9%→20.0%), `mjo_diff_column_mse` (5.9%→13.1%)。
- 同步重写了 4 处因 FDR 变化而失效的核心叙事：将 OLR 由“非主要预测因子”修正为“中等预测因子”；将扰动场 Column q 的结论由“极弱/无信号”修正为“独立显著的贡献”（32.4%）；更新了对“背景 vs 扰动信号强度对比”的总体概括。
- 使用脚本 `scripts/fix_report_docx_v4.py` 执行了最后的修正，确保逻辑与图表 100% 对齐。诊断图审计工作全面完成。
"""

with open(r'E:\Projects\ENSO_MJO_Tilt\AGENT_LOG.md', 'a', encoding='utf-8') as f:
    f.write(log_entry)
print("AGENT_LOG.md appended successfully.")
