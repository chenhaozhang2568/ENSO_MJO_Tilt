import os
from docx import Document

def fix_document(docx_path):
    doc = Document(docx_path)
    fixes = 0

    for i, p in enumerate(doc.paragraphs):
        old_text = p.text
        new_text = old_text
        fixed = False

        # P391: bg_diff_olr
        if 'FDR 显著比例 = 26.4%' in new_text and '⬆(弱)' in new_text:
            new_text = new_text.replace('⬆(弱)', '⬆⬆')
            fixed = True

        # P411: OLR prediction
        if 'OLR 不是主要预测因子' in new_text:
            new_text = new_text.replace('OLR 不是主要预测因子', 'OLR 是中等强度的预测因子（背景 FDR=39.6%）')
            fixed = True

        # P471: MJO column_q
        if '中心东侧（+30°~+60°）有弱正相关——扰动前方水汽更多时可能略快。但不通过严格的 FDR 阈值。' in new_text:
            new_text = new_text.replace(
                '中心东侧（+30°~+60°）有弱正相关——扰动前方水汽更多时可能略快。但不通过严格的 FDR 阈值。',
                '大面积正相关（FDR=32.4%），后方（-90°~-50°）和前方（+80°~+130°）均有显著正相关——MJO 扰动内部的水汽异常结构与其传播速度也有较强联系。'
            )
            fixed = True

        # P485: Column q perturbation signal
        if '扰动坐标下信号仍弱，再次确认"环境 > 内部结构' in new_text:
            new_text = new_text.replace(
                '扰动坐标下信号仍弱，再次确认"环境 > 内部结构',
                '扰动坐标下信号同样显著（MJO corr FDR=32.4%），说明 MJO 内部的水汽异常对速度也有显著独立贡献，但整体仍弱于背景场（51.4%）'
            )
            fixed = True

        # P621: Background vs Perturbation
        if '背景坐标的信号始终远强于扰动坐标。速度仍由环境控制。' in new_text:
            new_text = new_text.replace(
                '背景坐标的信号始终远强于扰动坐标。速度仍由环境控制。',
                '对于绝大多数变量，背景坐标的信号强于扰动坐标（除柱积分水汽外），表明速度主要由环境约束控制。'
            )
            fixed = True

        # P1127: Overall perturbation summary
        if '扰动场 FDR 很低（< 3.5%）' in new_text:
            new_text = new_text.replace(
                '扰动场 FDR 很低（< 3.5%）',
                '2D 扰动场 FDR 很低（q、w 等约 3%），但 1D 柱积分扰动场信号较强（Col q 达 32.4%）'
            )
            fixed = True

        if fixed:
            for run in p.runs:
                run.text = ''
            p.runs[0].text = new_text
            fixes += 1
            print(f'Fixed P{i}')
            print(f'  Old: {old_text[:80]}...')
            print(f'  New: {new_text[:80]}...')

    if fixes > 0:
        doc.save(docx_path)
        print(f'\nSuccess! Applied {fixes} fixes to the document.')
    else:
        print('\nNo fixes were applied. Could not find exact matching text.')

if __name__ == "__main__":
    report_path = r'C:\Users\Lenovo\Desktop\诊断图总结报告_修正版.docx'
    fix_document(report_path)
