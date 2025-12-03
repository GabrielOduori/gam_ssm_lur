# Response to Reviewers

**Manuscript:** Hybrid Generalized Additive-State Space Modelling for Urban NO₂ Prediction: Integrating Spatial and Temporal Dynamics

**Journal:** Environmental Modelling & Software

**Original Manuscript Number:** [Reference number]

**Authors:** Gabriel Oduori, Chiara Cocco, Payam Sajadi, Francesco Pilla

---

We thank the Editor and Associate Editor for the opportunity to revise and resubmit this manuscript. We have carefully addressed all comments and have substantially revised the paper. Below, we provide a point-by-point response to each comment, with the revised text shown in **boxed sections**.

---

## Response to Associate Editor

### General Comment 1

> *"The organization of topics in the paper could be improved. Care should be taken to make sure that topics are not 'reintroduced' or that statements within the paper are contradictory."*

**Response:** We have thoroughly reorganized the manuscript to eliminate redundancies and contradictions. Specifically:

- NO₂ and the study context are now introduced once in Section 1 (Introduction) and referenced thereafter without re-explanation
- Acronyms are defined on first use only (e.g., ESCAPE, LUR, SSM, GAM)
- The limitations of static LUR are stated once in Section 2.1, not repeated across sections
- Section 3 (Methods) has been restructured with clear subsections: Study Area → Data Sources → Modelling Framework → Parameter Estimation → Computational Scalability → Evaluation
- Bullet-point lists have been converted to flowing prose throughout

The revised manuscript now follows a logical progression: gap identification (§1) → literature positioning (§2) → methodology (§3) → results (§4) → discussion (§5).

---

### General Comment 2

> *"The framing of the modeling contribution is not currently very sophisticated. The authors seem to ignore the body of work on dynamic modeling in this area, and they would be well served to put their work in better context."*

**Response:** We have completely rewritten the Introduction and Section 2 to properly acknowledge and position our work relative to existing spatiotemporal LUR methods. We now explicitly discuss Ma et al. (2024)'s comprehensive review and the various approaches documented therein, clearly articulating what our hybrid framework offers beyond these existing methods.

**Revised Introduction (Paragraphs 3-4):**

> Over the past decade, LUR methodology has evolved considerably. Ma et al. (2024) provide a comprehensive review documenting this progression, from early cross-sectional models toward increasingly sophisticated spatiotemporal frameworks. Several strategies have been employed to incorporate temporal dynamics: developing separate models for different time periods (Masiol et al., 2018), including temporally varying covariates such as meteorological variables (Rahman et al., 2020), and applying machine learning methods capable of capturing complex spatiotemporal interactions (Wang et al., 2023). Generalised Additive Models (GAMs) have gained particular traction, as they accommodate non-linear predictor-response relationships while retaining the interpretability that distinguishes LUR from black-box alternatives (Hastie and Tibshirani, 1990; Hasenfratz et al., 2014; Lautenschlager et al., 2020).
>
> Despite these advances, existing spatiotemporal LUR approaches share a common limitation: they treat temporal variation through model structure (separate hourly models, time-indexed covariates) rather than as an explicit dynamical process. This distinction has important consequences. First, uncertainty is typically quantified cross-sectionally at each time point, without propagation of estimation error across the temporal sequence. Second, predictions at adjacent time steps are treated as conditionally independent given covariates, precluding the temporal smoothing that could improve estimates during periods of noisy or missing observations. Third, parameter estimation proceeds separately for spatial and temporal components, forgoing the efficiency gains available from joint inference. These limitations become acute when the objective shifts from retrospective exposure assessment toward real-time forecasting, where principled handling of temporal dynamics and associated uncertainties is critical.

---

### General Comment 3

> *"Results could be improved, as far as how the figures are made legible and understandable. More connection to the real-world case study and data would also help."*

**Response:** All figures have been regenerated with improved clarity:
- Increased font sizes for axis labels and legends
- Added colorbars with explicit units to all spatial plots
- Expanded figure captions to be self-contained (see responses to Specific Comment 13)
- Added panel labels (a), (b), (c), (d) for multi-panel figures
- Used colorblind-friendly palettes (viridis)

We have also strengthened the connection to the Dublin case study throughout the Results section, including specific references to geographic features (M50 motorway corridor, city centre), policy context (EU Ambient Air Quality Directive), and practical implications for exposure assessment.

---

## Response to Specific Comments

### Specific Comment 1.1

> *"The contribution relies on generic discussion of different vague modeling concepts (static LUR versus dynamic, state space models, etc.). More context of the contribution would be helpful, beyond simply saying that machine learning models are not interpretable. For example, are GAM-LUR models the standard of practice?"*

**Response:** We have revised the cover letter and Introduction to provide specific context. GAM-LUR is indeed increasingly adopted as a methodologically rigorous alternative to linear LUR, and we now cite key methodological papers establishing this.

**Revised Introduction (Paragraph 6):**

> The framework offers three principal contributions relative to existing spatiotemporal LUR methodology. First, it provides coherent uncertainty quantification: posterior distributions over latent pollution states incorporate both spatial covariate uncertainty and temporal process noise, yielding prediction intervals with demonstrably better calibration than static alternatives. Second, it enables temporal smoothing that separates signal from noise in high-frequency monitoring data, improving estimates during periods of sensor dropout or anomalous readings. Third, through adaptive matrix representations (dense for small networks, sparse block-diagonal for large ones), the approach scales to thousands of spatial locations while maintaining computational tractability—addressing a practical barrier that has limited SSM adoption in urban air quality applications.

---

### Specific Comment 1.2

> *"The novelty of the work should not really rest on implementing in Python or Numpy, which seem to be standards of practice in current times. Also it was curious that the authors say that the framework is 'sharable upon reasonable request' but also that it is 'open source' -- a contradiction. I also noted that the GitHub link is currently empty..."*

**Response:** We apologize for this oversight. The GitHub repository has now been fully populated with:
- Complete source code as a pip-installable Python package
- Comprehensive documentation and docstrings
- Test suite with pytest
- Example scripts including paper reproduction
- MIT License

We have removed claims about implementation novelty. The novelty rests on the *model formulation* (embedding LUR within an SSM framework) and *demonstrated performance*, not the implementation technology.

**Revised text (end of Introduction):**

> All code and data preprocessing workflows are publicly available at https://github.com/GabrielOduori/lur_space_state_model to facilitate replication.

**Revised text (Section 3.7 Implementation):**

> The complete framework was implemented in Python 3.10. Key dependencies include NumPy (Harris et al., 2020) for array operations, SciPy (Virtanen et al., 2020) for sparse matrices and numerical optimisation, pyGAM (Servén and Brummitt, 2018) for the GAM component, and Matplotlib (Hunter, 2007) for visualisation. SHAP (Lundberg and Lee, 2017) was used for feature importance analysis.
>
> Source code is available at https://github.com/GabrielOduori/lur_space_state_model. Processed data and intermediate outputs are archived on Zenodo (https://zenodo.org/uploads/16534138).

---

### Specific Comment 1.3

> *"A 'reduction in RMSE' is OK, but without proper context this is not helpful by itself. What is the RMSE being compared to?"*

**Response:** We now explicitly state the baseline in both the Abstract and Results.

**Revised Abstract:**

> Applied to hourly nitrogen dioxide (NO₂) observations from Dublin, Ireland, integrating regulatory monitors, TROPOMI satellite retrievals, and SCATS traffic data, the hybrid model achieves root mean square error of 0.53 µg/m³ compared to 1.41 µg/m³ for a static GAM-LUR baseline (62% reduction), improves the coefficient of determination from R² = −0.15 to R² = 0.84, and produces 95% prediction intervals with near-nominal coverage.

**Revised Introduction (Paragraph 7):**

> We demonstrate the framework using hourly NO₂ data from Dublin, Ireland, integrating observations from regulatory monitors, TROPOMI satellite retrievals, SCATS traffic counts, and OpenStreetMap land use features. Relative to a static GAM-LUR baseline fitted on identical covariates, the hybrid model achieves a 62% reduction in root mean square error (RMSE from 1.41 to 0.53 µg/m³) and improves the coefficient of determination from R² = −0.15 to R² = 0.84. Critically, the 95% prediction intervals achieve near-nominal coverage, whereas the static model systematically underestimates uncertainty.

---

### Specific Comment 2

> *"The highlights are inadequate ('A novel hybrid modelling framework' and 'Scalability and practical applications') -- they should somewhat stand on their own to explain the main points of the work."*

**Response:** We have completely rewritten the highlights to be self-contained and informative.

**Revised Highlights:**

> - Hybrid GAM-LUR-SSM framework integrates spatial land use regression with temporal state space dynamics
> - Expectation-Maximisation algorithm jointly estimates transition and noise covariance parameters
> - 62% RMSE reduction over static GAM-LUR baseline (0.53 vs 1.41 µg/m³) for Dublin NO₂
> - 95% prediction intervals achieve near-nominal coverage through principled uncertainty propagation
> - Block-diagonal Kalman filtering enables scalable inference for 8,700+ spatial locations
> - Open-source Python implementation with reproducible workflow

---

### Specific Comment 3

> *"In the abstract, the opening 'Models are useful...' is too vague for our journal -- which is all about modeling! The sentence that starts 'This paper introduces...' could be a better opening..."*

**Response:** We have rewritten the Abstract to open with the specific methodological gap rather than a generic statement.

**Revised Abstract:**

> Land Use Regression (LUR) models are widely used for urban air pollution mapping, yet existing spatiotemporal extensions treat temporal variation through model structure—separate hourly models, time-indexed covariates—rather than as an explicit dynamical process. This limits their capacity for uncertainty propagation, temporal smoothing, and principled forecasting. We propose a hybrid framework that integrates Generalised Additive Model (GAM)-based LUR with linear Gaussian State Space Models (SSMs) to address these limitations. The GAM component captures persistent spatial heterogeneity through smooth functions of land use, road network, and traffic covariates. Residuals from this spatial model are then modelled as a latent dynamical process via an SSM, with parameters estimated jointly using an Expectation-Maximisation algorithm and inference performed through Kalman filtering and Rauch-Tung-Striebel smoothing. Adaptive matrix representations—dense, sparse, or block-diagonal depending on network size—enable scalable inference for thousands of spatial locations. Applied to hourly nitrogen dioxide (NO₂) observations from Dublin, Ireland, integrating regulatory monitors, TROPOMI satellite retrievals, and SCATS traffic data, the hybrid model achieves root mean square error of 0.53 µg/m³ compared to 1.41 µg/m³ for a static GAM-LUR baseline (62% reduction), improves the coefficient of determination from R² = −0.15 to R² = 0.84, and produces 95% prediction intervals with near-nominal coverage. The framework offers coherent uncertainty quantification, optimal temporal smoothing, and computational tractability, advancing spatiotemporal LUR methodology for urban air quality assessment. Code and data are openly available.

---

### Specific Comment 4

> *"In general, all acronyms should be defined, such as ESCAPE on page 2."*

**Response:** All acronyms are now defined on first use. ESCAPE is defined in the revised Introduction.

**Revised text:**

> Originally introduced by Briggs et al. (1997) and subsequently refined in large-scale studies such as ESCAPE (European Study of Cohorts for Air Pollution Effects; Beelen et al., 2013), LUR models relate monitored pollutant concentrations to geographic predictors—road proximity, land use classifications, population density—via regression.

---

### Specific Comment 5

> *"At the end of page 2, I would suggest avoiding the style of bullet points so popular with ChatGPT and other LLMs. The first bullet: 'Preserves interpretability: GAMs retain...' is odd..."*

**Response:** We have removed all bullet-point lists from the Introduction and converted them to flowing prose. The contributions are now presented as a cohesive paragraph.

**Revised Introduction (Paragraph 6):**

> The framework offers three principal contributions relative to existing spatiotemporal LUR methodology. First, it provides coherent uncertainty quantification: posterior distributions over latent pollution states incorporate both spatial covariate uncertainty and temporal process noise, yielding prediction intervals with demonstrably better calibration than static alternatives. Second, it enables temporal smoothing that separates signal from noise in high-frequency monitoring data, improving estimates during periods of sensor dropout or anomalous readings. Third, through adaptive matrix representations (dense for small networks, sparse block-diagonal for large ones), the approach scales to thousands of spatial locations while maintaining computational tractability—addressing a practical barrier that has limited SSM adoption in urban air quality applications.

---

### Specific Comment 6

> *"The introduction's treatment of spatiotemporal modeling is severely lacking. The authors say 'Whilst different flavours of LUR methods exist [4]' they seem to argue that the static models are missing the dynamic aspect. But reference [4] includes multiple categories of spatiotemporal modeling..."*

**Response:** This is an important point and we thank the reviewer for highlighting it. We have substantially revised the Introduction and Section 2 to properly acknowledge the existing spatiotemporal LUR literature, including the comprehensive taxonomy in Ma et al. (2024).

**Revised Section 2.1 (From Static to Spatiotemporal Land Use Regression):**

> Land Use Regression emerged in the 1990s as a cost-effective alternative to dense monitoring networks for characterising intra-urban air pollution variability (Briggs et al., 1997). The core premise is straightforward: pollutant concentrations measured at a limited number of locations can be related to spatially referenced predictors—road proximity, traffic intensity, land use composition, population density—via regression, enabling prediction across unmonitored sites. The ESCAPE project (Beelen et al., 2013) established standardised protocols that have since been widely adopted, and Hoek et al. (2008) provide an authoritative early review of model components and performance.
>
> Classical LUR models are inherently cross-sectional: they estimate time-averaged concentrations (typically annual means) and assume that predictor-response relationships remain constant over the averaging period. This static formulation is adequate for long-term exposure assessment in epidemiological cohorts but poorly suited to applications requiring finer temporal resolution—short-term health impact studies, real-time public advisories, or evaluation of transient interventions such as traffic restrictions.
>
> Recognising this limitation, researchers have pursued several strategies to incorporate temporal dynamics into the LUR framework. Ma et al. (2024) categorise these approaches in their comprehensive review spanning 2011–2023:
>
> *Temporally stratified models.* The most direct extension involves fitting separate LUR models for different time periods. Masiol et al. (2018) developed 24 distinct hourly PM models, each with 16–26 predictors and R² values ranging from 0.63 to 0.77. Don et al. (2013) compared hourly black carbon models using static versus dynamic covariates, concluding that independent hourly models outperformed pooled approaches with dummy variables. While effective, this strategy multiplies computational burden, prohibits information sharing across time points, and provides no mechanism for temporal smoothing or uncertainty propagation.
>
> *Time-varying covariates.* An alternative approach retains a unified model structure but incorporates predictors that vary temporally—meteorological variables, satellite retrievals, or real-time traffic counts. Rahman et al. (2020) achieved R² values of 0.64–0.88 for hourly particle number concentrations in Brisbane using Random Forest with dynamic meteorological inputs. Wang et al. (2023) integrated TROPOMI satellite data with ground observations in a geostatistical ST-LUR framework for Shanghai, estimating daily Air Quality Index at 100-metre resolution. These models capture covariate-driven temporal variation but treat residual temporal structure as noise rather than signal.
>
> *Machine learning methods.* Flexible algorithms including Random Forest, gradient boosting, and neural networks can implicitly learn complex spatiotemporal interactions when provided appropriate feature engineering (Yang et al., 2018; Xu et al., 2019). However, their black-box nature limits interpretability—a significant drawback when policy applications require understanding *why* concentrations vary, not merely predicting *that* they vary. Hybrid approaches combining machine learning with geostatistical methods (e.g., residual kriging) partially address spatial correlation but typically do not model temporal dynamics explicitly (Wu et al., 2018).
>
> *Geostatistical extensions.* Spatiotemporal kriging and its variants model correlation structures across both space and time, often combined with LUR mean functions (Sampson et al., 2011; Xu et al., 2019). These approaches can interpolate and smooth observations but require specification of space-time covariance functions that may not align with the mechanistic drivers of pollution dynamics. Computational costs scale poorly with the number of observations, limiting applicability to dense sensor networks.
>
> Despite this methodological diversity, a common thread runs through existing spatiotemporal LUR approaches: temporal variation is accommodated through model *structure* (separate models, time-indexed covariates, flexible learners) rather than treated as an explicit *dynamical process*. This distinction carries three practical consequences that motivate the present work:
>
> First, **uncertainty quantification remains cross-sectional**. Prediction intervals at each time point reflect estimation error in model parameters and residual variance but do not account for how uncertainty evolves and propagates through time. When observations are missing or anomalous, static models have no principled basis for borrowing strength from adjacent time points.
>
> Second, **temporal smoothing is implicit rather than principled**. Time-varying covariate models smooth predictions indirectly through the temporal autocorrelation of their inputs (e.g., meteorology), but this smoothing is not optimised for the pollution process itself. Separate hourly models forgo smoothing entirely, treating each time slice as independent.
>
> Third, **parameter estimation is fragmented**. Fitting separate models for each time period discards information about the underlying process stability; pooled models with time covariates confound spatial and temporal effects. Neither approach exploits the sequential structure of the data for efficient joint inference.

---

### Specific Comment 7

> *"In my opinion, section 2.1 is too general and is not adding value. Again, if spatio-temporal LUR models already exist, the focus of the background section in this paper should be to address the *existing* spatiotemporal LUR and then put into context why the authors' new method is a good contribution."*

**Response:** We have completely rewritten Section 2. The original general regression primer has been removed. The new Section 2.1 (shown above) now directly engages with existing spatiotemporal LUR approaches and clearly articulates the gaps our method addresses.

**New Section 2.3 (Bridging the Gap: Motivation for a Hybrid Framework):**

> The preceding review reveals complementary strengths and limitations in the two modelling traditions. LUR methods excel at capturing spatial heterogeneity through interpretable covariate relationships but lack principled mechanisms for temporal dynamics, uncertainty propagation, and adaptive smoothing. SSMs provide exactly these capabilities but have been applied primarily to aggregate time series without the spatial granularity that urban air quality assessment demands.
>
> A hybrid framework integrating GAM-based LUR with State Space modelling can leverage the strengths of both approaches. The GAM component captures the persistent spatial structure attributable to land use, road networks, and other geographic features—the "fixed" component of the pollution field that varies across space but remains relatively stable over short time horizons. The SSM component then models the temporal evolution of residuals: the deviations from land-use-explained concentrations driven by traffic fluctuations, meteorological variation, and other time-varying factors that the spatial model cannot anticipate.
>
> This decomposition offers several advantages over existing spatiotemporal LUR approaches:
>
> **Principled uncertainty quantification.** The SSM provides posterior distributions over latent states at each time point, with covariances that propagate through the Kalman recursions. Prediction intervals naturally widen during data gaps and narrow as observations accumulate, reflecting the actual information content of the monitoring record.
>
> **Optimal temporal smoothing.** The Rauch-Tung-Striebel smoother produces state estimates that optimally balance fidelity to observations against process model constraints. Noisy measurements are attenuated; missing values are imputed using temporal context. This smoothing is principled in the sense of minimising mean squared error under the assumed model.
>
> **Joint parameter estimation.** The EM algorithm estimates transition dynamics, process noise, and observation noise simultaneously, exploiting the full spatiotemporal structure of the data. Information flows across both time (through the Kalman recursions) and space (through the shared model parameters), improving efficiency relative to fragmented estimation strategies.
>
> **Scalability.** While full SSM inference scales cubically with state dimension—a prohibitive cost for large sensor networks—structured approximations (diagonal covariances, block decompositions, sparse matrix representations) can reduce complexity to near-linear scaling, making the approach practical for urban-scale applications with thousands of spatial locations.

---

### Specific Comment 8

> *"Page 7 has another set of bullet points that do not flow with the rest of the text. The authors should look through the paper and reduce the number of times they mention that 'LUR methods cannot account for time-varying phenomena'; this bullet list also brings up the TROPOMI issue, which the readers would not really understand at this point."*

**Response:** The bullet points on page 7 have been removed and integrated into the flowing prose of Section 2.1 (shown above). The statement about LUR limitations now appears once, in proper context. TROPOMI is now introduced in Section 3.2.2 (Data Sources) with appropriate explanation before being referenced elsewhere.

**Revised Section 3.2.2 (Satellite NO₂ Observations):**

> The TROPOspheric Monitoring Instrument (TROPOMI) aboard Sentinel-5P provides daily observations of tropospheric NO₂ vertical column density (VCD) at approximately 5.5 × 3.5 km resolution (Griffin et al., 2019). TROPOMI's sun-synchronous orbit yields overpasses between 11:00–15:00 local time; at Dublin's latitude (~53.3°N), partial swath overlap occasionally permits two valid retrievals per day.
>
> Level-2 NO₂ products were accessed via the Google Earth Engine API. For each grid cell centroid, we extracted cloud-screened VCD values (mol/m²) and converted to near-surface concentration estimates (µg/m³) following Savenets (2021):
>
> C = (C_col / H) × M × A
>
> where C_col is the column density, H is the effective mixing layer height (obtained from ERA5 reanalysis), M = 46.01 g/mol is the molar mass of NO₂, and A = 1000 is a unit conversion factor. We acknowledge that this conversion introduces uncertainty, particularly under conditions of strong vertical stratification; however, the satellite-derived values serve primarily as a spatially extensive covariate rather than the target variable.

---

### Specific Comment 9

> *"On page 8, the authors re-introduce the concept that they are looking at NO2, which has already been mentioned many times. Many 're-introductions' exist; for example, acronyms are re-defined on page 18."*

**Response:** We have carefully reviewed the entire manuscript and removed all re-introductions. NO₂ is introduced once in Section 1 (Introduction) with its health relevance, and referred to simply as "NO₂" thereafter. All acronyms are defined on first use only—we have removed duplicate definitions on page 18 and elsewhere.

---

### Specific Comment 10

> *"The concept of feature selection (section 3.6) could also be framed as part of the study's contribution in the introduction."*

**Response:** We now mention the multi-stage feature selection pipeline as a methodological element in the Introduction, and have repositioned it more prominently in Section 3.

**Added to Introduction (Paragraph 7):**

> The framework incorporates a structured feature selection pipeline—combining correlation-based filtering, variance inflation factor screening, and Random Forest importance ranking—to address the high-dimensional covariate space typical of LUR applications.

**Revised Section 3.3.2 (Feature Selection):**

> Given the high-dimensional predictor space (291 candidate variables), a structured feature selection pipeline was applied prior to GAM fitting to mitigate multicollinearity and improve interpretability. The pipeline comprised three stages:
>
> **Stage 1: Correlation-based filtering.** For each pair of predictors with absolute Pearson correlation exceeding 0.8, the variable with lower variance was removed. This step reduced redundancy among spatially correlated buffer-distance features.
>
> **Stage 2: Variance Inflation Factor (VIF) screening.** Remaining predictors were evaluated for multicollinearity via VIF (Hair et al., 2019). Variables with VIF > 10 were iteratively removed until all retained predictors satisfied the threshold.
>
> **Stage 3: Importance-based selection.** A Random Forest regressor (Breiman, 2001) was trained on the reduced predictor set, and feature importances were extracted. The top 30 predictors by importance were retained for the final GAM, supplemented by domain-informed "force-keep" variables (traffic volume, motorway proximity) to ensure policy-relevant covariates remained in the model regardless of their data-driven ranking.
>
> This pipeline reduced the predictor set from 291 to 56 variables after correlation and VIF filtering, and to 15 variables in the final GAM specification (Table 1).

---

### Specific Comment 11

> *"I'm not sure how useful Table 1 is by itself; perhaps visualization or analysis of the input data would be helpful to show."*

**Response:** We have added SHAP (SHapley Additive exPlanations) visualization as Figure 3 to complement Table 1. The SHAP summary plot shows both feature importance rankings and the direction/magnitude of effects, providing richer insight than the table alone.

**Revised Figure 3 caption:**

> **Figure 3: Feature importance analysis using SHAP values.** Bee swarm plot showing the contribution of each predictor to NO₂ predictions in the GAM-LUR model. Features are arranged vertically in descending order of mean absolute SHAP value (most important at top). Each point represents one observation; horizontal position indicates the SHAP value (impact on model output in µg/m³), with positive values (right) indicating increased predicted NO₂ and negative values (left) indicating decreased predictions. Point colour encodes the original feature value: red indicates high values, blue indicates low values. Key findings: motorway proximity and density within 1,000 m are the strongest predictors; industrial and residential land use at 1,050 m buffers show moderate importance; traffic volume from SCATS detectors contributes less than static road network features, suggesting that spatial configuration dominates over temporal traffic variation in explaining NO₂ levels.

---

### Specific Comment 12

> *"On the bottom of page 19, a figure seems to be missing?"*

**Response:** We apologize for this error, which was a LaTeX cross-reference issue. The missing figure reference has been corrected and all figure cross-references have been verified.

---

### Specific Comment 13

> *"It is difficult to interpret what figure 4 is attempting to show, and the caption could be improved. The same comment applies to the other figures, such as figure 6, where the colors are not clearly defined, the legends are hard to read, and the caption is uninformative."*

**Response:** All figure captions have been substantially expanded to be self-contained and informative. Figures have been regenerated with larger fonts, explicit color definitions, and panel labels.

**Revised Figure 4 caption:**

> **Figure 4: Temporal evolution of observed and smoothed NO₂ concentrations at six representative grid locations.** Each panel displays the time series for one spatial location over the 50-day analysis period (x-axis: days; y-axis: standardised NO₂ residuals after removing the GAM spatial mean). Black points show observed values; solid blue lines show Kalman-smoothed state estimates; shaded blue bands indicate 95% posterior credible intervals. Location IDs (top of each panel) correspond to grid cell indices in the spatial domain. The smoothed estimates filter out high-frequency noise while preserving systematic temporal patterns. Credible intervals widen during periods of missing data and narrow when observations are available, demonstrating the model's adaptive uncertainty quantification. Note the common diurnal and weekly patterns across locations, reflecting shared temporal drivers (traffic cycles, meteorology) captured by the state space dynamics.

**Revised Figure 5 caption:**

> **Figure 5: Spatial distribution of NO₂ concentrations across three time points.** Top row: observed NO₂ residuals (after removing spatial mean) at Day 0 (left), Day 13 (centre), and Day 26 (right). Bottom row: corresponding Kalman-smoothed estimates for the same days. Colour scale indicates NO₂ concentration anomaly (µg/m³ relative to the GAM-predicted spatial mean), with warmer colours (yellow–red) denoting positive anomalies (higher than land-use-predicted levels) and cooler colours (blue) denoting negative anomalies. Axes show geographic coordinates (longitude, latitude in decimal degrees). The observed fields (top) exhibit considerable spatial noise; the smoothed fields (bottom) reveal coherent spatial structure while attenuating measurement error. Notable features include elevated concentrations along the M50 motorway corridor (western edge) and in the city centre, consistent with traffic emission patterns.

**Revised Figure 6 caption:**

> **Figure 6: Expectation-Maximisation algorithm convergence diagnostics.** Four panels tracking the EM estimation procedure across iterations (x-axis). **Top left:** Log-likelihood values (×10⁶) showing monotonic increase and convergence after approximately five iterations, consistent with theoretical EM properties. **Top right:** Evolution of key parameter traces—tr(**T**) (state transition matrix trace, blue), tr(**Q**) (process noise covariance trace, red), and tr(**R**) (observation noise covariance trace, green)—demonstrating stabilisation of all parameters. **Bottom left:** Log-likelihood increment (absolute change between successive iterations) on logarithmic scale, declining below the 10⁻⁶ convergence threshold by iteration 5. **Bottom right:** Parameter change magnitudes |Δtr(**T**)|, |Δtr(**Q**)|, and |Δtr(**R**)| showing rapid decay. These diagnostics confirm successful convergence of the EM algorithm and stability of the final parameter estimates.

**Revised Figure 7 caption:**

> **Figure 7: Model performance diagnostics comparing observed and smoothed estimates.** Four panels evaluating the hybrid GAM-SSM model. **Top left:** Scatter plot of observed versus Kalman-smoothed NO₂ residuals across all space-time points. Pearson correlation r = 0.946 (annotated) indicates strong agreement; dashed red line shows the 1:1 reference. **Top right:** Temporal variance comparison showing observed variance (black) and smoothed variance (blue) over the 50-day period; variance reduction demonstrates effective noise filtering while preserving signal dynamics. **Bottom left:** Spatial variance comparison; each point represents one grid location, with observed spatial variance (x-axis) plotted against smoothed spatial variance (y-axis). Points below the 1:1 line indicate variance reduction through smoothing; the close alignment suggests spatial structure is preserved. **Bottom right:** Spatial distribution of location-specific RMSE (colour scale: 0.4–0.8 µg/m³). Higher errors concentrate in the city centre and along major arterials, likely reflecting greater NO₂ variability in high-traffic areas that challenges even the hybrid model.

**Revised Figure 8 caption:**

> **Figure 8: Residual diagnostics for the hybrid GAM-SSM model.** Six panels assessing model adequacy. **Top left:** Residuals (observed minus smoothed) versus fitted values; random scatter around zero with no systematic pattern indicates absence of bias and heteroscedasticity. **Top centre:** Quantile-quantile (Q-Q) plot comparing residual distribution to theoretical normal quantiles; close adherence to the diagonal confirms approximate normality, with minor deviation in the upper tail suggesting slight positive skewness. **Top right:** Histogram of residuals with overlaid normal density curve (red); the empirical distribution closely matches the theoretical, supporting the Gaussian assumption. **Bottom left:** Temporal residual pattern showing daily mean residuals (blue line) with ±1 standard deviation bands (grey shading); residuals fluctuate around zero throughout the 50-day period, though excursions during days 15–20 and 35–40 suggest episodic model underperformance potentially linked to unmeasured meteorological events. **Bottom centre:** Spatial residual distribution across the study domain; absence of systematic clustering indicates no unmodelled spatial structure. **Bottom right:** Residual autocorrelation function (ACF) for lags 0–50; correlations remain within acceptable bounds (±0.05, dashed lines) for all lags, confirming that temporal dependencies have been adequately captured by the state space component.

---

## Summary of Changes

| Section | Change |
|---------|--------|
| **Highlights** | Completely rewritten to be self-contained and quantitative |
| **Abstract** | New opening addressing specific gap; added baseline comparison |
| **Introduction** | Rewritten to acknowledge ST-LUR literature; contributions in prose |
| **Section 2** | Removed basic primer; added comprehensive ST-LUR review; clear gap statement |
| **Section 3** | Restructured; TROPOMI properly introduced; feature selection elevated |
| **All Figures** | Regenerated with larger fonts, colorbars, panel labels |
| **All Captions** | Expanded to be self-contained with panel-by-panel descriptions |
| **GitHub** | Repository fully populated with code, tests, documentation |
| **Throughout** | Removed redundancies, duplicate acronym definitions, bullet points |

---

We believe these revisions substantially address all reviewer concerns. We thank the Editor and reviewers for their constructive feedback, which has significantly improved the manuscript.

Sincerely,

Gabriel Oduori, Chiara Cocco, Payam Sajadi, Francesco Pilla
University College Dublin
