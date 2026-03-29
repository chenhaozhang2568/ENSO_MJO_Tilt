# ENSO_MJO_Tilt Plotting Standards

## 1. Purpose

This document defines the plotting standards for `E:\Projects\ENSO_MJO_Tilt`.
The goal is to make figures:
- scientifically defensible,
- visually consistent,
- comparable across scripts,
- easy to reuse in notes, slides, and papers.

These rules apply to all newly created or revised figures unless there is a clear reason to deviate.
If a figure intentionally breaks a rule, state the reason in the script comments or the final summary.

## 2. Overall Principles

1. Prioritize interpretability over decoration.
2. Use the same visual rule for the same scientific meaning across the repository.
3. Keep figure titles descriptive, not argumentative.
4. Separate visual facts from statistical claims.
5. Prefer fewer, clearer annotations over dense overlays.
6. Preserve comparability across related figures by fixing axis ranges and color limits whenever comparison is the point.

## 3. Figure Classes

Use one of these classes and keep conventions stable within each class.

### 3.1 Spatial field figures
Examples:
- lat-lon correlation maps
- lat-lon mean/composite maps
- longitude-pressure sections
- longitude-height sections

Typical elements:
- filled contour or pcolormesh field
- optional significance stippling
- one colorbar
- axis labels with units or coordinate notation

### 3.2 Statistical comparison figures
Examples:
- boxplot by ENSO phase
- violin / box / scatter combinations
- grouped comparisons with p-value brackets

Typical elements:
- group colors fixed across the repository
- sample size shown in x tick labels or caption
- effect size or p-value handled consistently

### 3.3 Relationship figures
Examples:
- scatter plots
- regression plots
- predicted vs actual
- ONI vs tilt / speed

Typical elements:
- axis labels with units
- fitted line only if justified
- r, p, or model score shown in a consistent position

### 3.4 Summary figures
Examples:
- heatmaps of max absolute correlation
- ranking bars
- multi-panel synthesis figures

Typical elements:
- compact labels
- consistent color semantics
- limited text inside panels

## 4. File Output Standards

### 4.1 Format
- Default output format: `png`
- Final publication-ready or slide-ready figure, if needed later: also save `pdf`
- Do not save only `jpg`

### 4.2 Resolution
- Exploratory figures: `dpi=180` or `dpi=200`
- Figures intended for summary, report, or publication draft: `dpi=300`
- Use one DPI value within the same script; do not mix 150 and 200 in one analysis family

### 4.3 Save behavior
- Always use `bbox_inches="tight"`
- Use explicit output directories by analysis family
- Do not overwrite outputs from a different method branch unless intended

### 4.4 File naming
Use lowercase snake_case and keep the structure:
- `mean_<var>.png`
- `corr_<var>.png`
- `diff_<var>.png`
- `composite_<var>.png`
- `summary_<theme>.png`

If levels are present, append pressure level explicitly:
- `corr_q_850hpa.png`
- `diff_t_500hpa.png`

Do not mix naming styles such as `camelCase`, spaces, or ambiguous suffixes like `_new_final2`.

## 5. Typography And Layout

### 5.1 Fonts
- Default sans-serif: `DejaVu Sans`
- Fallback: `Arial`
- Avoid mixing Chinese and English fonts within one figure unless necessary
- Do not use decorative fonts

### 5.2 Font sizes
Use these as defaults:
- figure title: 13-15
- subplot title: 11-13
- axis label: 11-12
- tick label: 9-10
- annotation text: 8-10
- colorbar label: 10-11

### 5.3 Title rules
- Title should state what is plotted, not what is concluded
- Good: `Corr(OLR, Phase Speed)`
- Good: `Phase Speed by ENSO Phase`
- Avoid: `Strong evidence that ENSO controls MJO propagation`

### 5.4 Multi-panel layout
- Use shared sizing within the same figure family
- Keep panel spacing tight but readable
- Use `(a)`, `(b)`, `(c)` for paper-style multi-panel figures when panels will be cited in text
- Use a common colorbar whenever panels represent the same variable under comparable ranges

## 6. Color Standards

### 6.1 Diverging variables
Use diverging colormaps for anomalies, differences, and correlations:
- default: `RdBu_r`
- center at zero with `TwoSlopeNorm`
- make color limits symmetric around zero unless there is a strong reason not to

Applies to:
- correlation maps
- anomaly maps
- group differences
- vertical motion, temperature, humidity anomalies when plotted as signed departures

### 6.2 Sequential variables
Use sequential colormaps for strictly nonnegative magnitudes or frequencies.
Do not use `RdBu_r` if zero is not a meaningful center.

### 6.3 Rainbow prohibition
- Do not use rainbow colormaps such as `jet`
- Do not use visually misleading non-monotonic colormaps

### 6.4 Cross-figure comparability
If comparing the same variable across multiple groups, coordinates, or branches:
- use the same colormap
- use the same sign convention
- use the same `vmin/vmax` when direct visual comparison is intended

If color limits differ intentionally, state that explicitly in the title, caption, or script comments.

## 7. Axis Standards

### 7.1 Longitude
Use one of two explicit conventions and do not mix them within one figure family.

- Absolute longitude maps:
  - preferred display: `Longitude (буE)`
  - if data span `-180 to 180`, still label clearly and keep tick labels interpretable
- Relative longitude composites:
  - use `Relative Longitude (бу)`
  - zero must represent the convection-center reference point

Do not leave longitude sign convention implicit.

### 7.2 Latitude
- Label as `Latitude (буN)` when using signed latitude values
- Southern Hemisphere values can remain negative; no need to relabel as `буS` unless formatting code is intentionally added

### 7.3 Pressure axis
For pressure-coordinate vertical sections:
- invert the pressure axis so large pressure is at the bottom and small pressure is at the top
- label as `Pressure (hPa)`
- preferred ticks: `1000, 850, 700, 500, 300, 200`

### 7.4 Time axis
- For event-day figures, use either calendar date or event-relative day, not both unless necessary
- If event-relative day is used, label clearly as `Day since event start` or equivalent

### 7.5 Units
- Every quantitative axis and every colorbar must have units unless the quantity is dimensionless
- For correlations, label colorbar as `Pearson r`
- For normalized fields, indicate normalization explicitly if it matters for interpretation, e.g. `/ amp`

## 8. Significance Standards

This is the most important standard for this repository.

### 8.1 Default threshold
- Default significance threshold: `p < 0.05`
- Any other threshold must be explicitly stated in the figure or script

### 8.2 Interpretation rule
- Significance marks indicate where the tested null hypothesis is rejected under the chosen test
- They do not by themselves justify causal language
- They do not replace reporting effect size or field magnitude

### 8.3 Spatial maps: default marking rule
For lat-lon or lon-pressure fields:
- use stippling or sparse point overlay for significance
- default stipple color: black
- default marker: `.` or very small filled circle
- points must remain visually secondary to the main field
- reduce point density if the field becomes unreadable

Default rule for this repository:
- use black stippling for generic significance masks
- do not mix red and blue significance dots unless the dot color itself encodes a second quantity such as the sign of correlation
- if dot color encodes sign, state that clearly in the legend

### 8.4 Spatial maps: density control
Current figures show a tendency toward over-dense stippling.
To control this:
- if more than about 30-40 percent of valid grid cells are significant, prefer thinner or subsampled stippling
- alternatively use contour outlines for significant regions when dense stippling obscures the field
- do not allow significance markers to dominate the plot visually

### 8.5 Spatial maps: title policy
Do not place long significance statistics in every title by default.

Preferred:
- short title in the panel
- detailed significance summary in caption, console output, or accompanying text

Allowed when useful:
- brief suffix such as `p<0.05 stippling`

Avoid as default:
- `Sig(p<0.05): 1119/2448 (45.7%)` in every map title

### 8.6 Boxplots and grouped statistical figures
For group comparisons:
- use explicit brackets only for pre-specified comparisons
- default pair set for three-group ENSO plots:
  - El Nino vs Neutral
  - Neutral vs La Nina
  - El Nino vs La Nina
- bracket text should report exact p-value when possible
- stars may be included, but exact p-value should remain primary

Recommended formatting:
- `p=0.008`
- optional: `p=0.008 **`

Avoid:
- stars only with no p-value
- too many stacked brackets that consume the figure

### 8.7 Multiple testing
For large fieldwise testing or many repeated comparisons:
- note in the script whether p-values are raw or corrected
- if no multiple-comparison correction is applied, do not imply strict pointwise significance proves a robust spatial mechanism
- for summary interpretation, prefer phrases like `pointwise p<0.05` rather than `significant region` unless the testing framework justifies it

## 9. Statistical Figure Standards

### 9.1 Group colors
Fix ENSO colors across the repository:
- El Nino: `#E74C3C`
- Neutral: `#95A5A6`
- La Nina: `#3498DB`

Do not reassign these colors in later scripts.

### 9.2 Boxplots
- overlay jittered raw points when sample size is moderate
- keep jitter width modest and deterministic if reproducibility matters
- show median clearly
- outliers may remain visible, but should not dominate the figure

### 9.3 Sample size
Show sample size in x tick labels or caption for grouped statistical plots.
This is already a strength in current ENSO plots and should remain standard.

### 9.4 Regression and scatter plots
- show a regression line only if it adds analytical value
- annotate with `r`, `p`, and optionally `R^2` in a consistent corner
- keep marker alpha below 0.8 when points overlap
- use small to medium marker size

## 10. Heatmap Standards

For summary heatmaps like `summary_heatmap.png`:
- use diverging color centered at zero when values are signed
- print exact values in cells if the matrix is small enough
- use stars only as secondary significance annotation
- keep row labels short and scientifically meaningful
- avoid excessive precision; three decimals is enough for `r`

## 11. Variable Naming Standards In Figures

Use readable scientific labels.

Preferred examples:
- `OLR`
- `Specific Humidity (q)`
- `Vertical Velocity (omega)`
- `Phase Speed (m/s)`
- `Tilt_q (deg)`

Avoid raw internal variable names in final figures if a readable label is available.
For example, prefer `Vertical Velocity (omega)` over a symbol that may render inconsistently across environments.

## 12. What To Standardize Immediately In This Repository

These are the first issues to clean up because they already appear in current outputs.

1. Standardize DPI to `200` for exploratory figures and `300` for final figures.
2. Remove overly long significance counts from most map titles.
3. Keep significance stippling sparse enough that the field remains readable.
4. Fix one repository-wide convention for colorbar labels and units.
5. Keep ENSO group colors unchanged across every grouped figure.
6. Use `RdBu_r` plus symmetric limits for signed maps by default.
7. In vertical sections, always invert pressure axis.
8. Use consistent figure sizes within each analysis family.

## 13. Minimum Pre-Save Checklist

Before saving a figure, confirm all of the following:
- What is the one-sentence purpose of this figure?
- Are axis labels complete and unit-aware?
- Is the color scale appropriate for the variable type?
- Is zero centered when plotting a signed quantity?
- Are significance markings visible but not dominant?
- Does the title describe the plot without overstating its meaning?
- If comparing panels, are axis and color ranges truly comparable?
- Is the file name explicit enough to identify variable and context?

## 14. Recommended Next Implementation Step

To make these standards executable rather than aspirational:
- create `src/plot_style.py`
- centralize `rcParams`, standard figure sizes, color choices, axis formatters, colorbar helpers, and significance overlay helpers
- refactor new plotting scripts to call shared helper functions instead of duplicating style logic
