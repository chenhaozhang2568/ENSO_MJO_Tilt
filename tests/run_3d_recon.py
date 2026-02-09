# -*- coding: utf-8 -*-
"""
run_3d_recon.py: 独立运行 3D 重构（保留纬度）的脚本

Run:
    python E:\\Projects\\ENSO_MJO_Tilt\\tests\\run_3d_recon.py
"""

import sys
sys.path.insert(0, r"E:\Projects\ENSO_MJO_Tilt\src")

# Import directly - 02_mvEOF.py has been modified to include reconstruct_era5_fields_3d()
exec(open(r"E:\Projects\ENSO_MJO_Tilt\src\02_mvEOF.py", encoding="utf-8").read())

if __name__ == "__main__":
    reconstruct_era5_fields_3d()
