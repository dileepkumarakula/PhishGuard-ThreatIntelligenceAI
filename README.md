# PhishGuard

PhishGuard is a local React/Vite frontend and FastAPI service for safely analyzing a public HTTP(S) URL. The backend validates every destination, performs a bounded static fetch without running JavaScript, and selects one of two separate classifiers based on the evidence available.

## Analysis modes

- **Full webpage** (`analysis_mode: "full_webpage"`, `analysis_status: "complete"`): the preserved PhiUSIIL XGBoost pipeline is eligible only when all 54 original inputs are present and valid. The current live extractor does not assign at least 17 required dataset fields, so it cannot currently produce a complete 54-feature input from a live page. The full model and artifacts are preserved, but end-to-end full webpage prediction is not currently demonstrated.
- **URL only** (`analysis_mode: "url_only"`, `analysis_status: "url_only"`): the separately trained XGBoost pipeline uses 34 deterministic features derived from the submitted URL. The UI identifies the model and reports its 34 inputs; this is not a full webpage analysis.
- **Fallback error:** when the 54-feature model cannot run and the validated URL-only model is unavailable, the API returns an explicit error with no classification. HTTP 401/403 do not expose or analyze restricted page content; the URL-only model may still classify the submitted URL and labels the output URL-only. Unsafe targets are rejected.

For page-fetch failures, including DNS/connection errors, timeouts, HTTP 401/403/404/429/server errors, non-HTML pages, and empty or oversized HTML, the service can return a clearly labeled URL-only analysis if the URL model is loaded. For 401/403 it does not retry, authenticate, or inspect denied content. URL-derived analysis is separate and makes no claim of webpage coverage.

The original 54-feature model and artifacts are unchanged. The second model and its evaluation are kept separately at `backend/models/phishguard_url_model.joblib` and `backend/models/phishguard_url_model_evaluation.json`.

## Architecture

- `src/`: React UI, API client and local scan history. Each saved prediction records whether it used URL-only or full webpage mode.
- `backend/main.py`: FastAPI routes, artifact/schema checks, model selection, inference, readiness reporting, and configurable per-process `/predict` rate limiting.
- `backend/feature_extraction.py`: URL validation, public-IP-pinned retrieval, redirect checks and HTML parsing.
- `backend/url_features.py`: deterministic 34-feature URL-only schema shared by the trainer and live API.
- `backend/train_url_model.py`: dataset audit, same-fold grouped XGBoost baseline/candidate comparison, evaluation and separate URL-model training.
- `backend/models/`: original supplied model/schema artifacts plus the separate URL model and report.

The frontend never reads model artifacts. The backend serves API documentation at `/docs` and OpenAPI at `/openapi.json`.

## Preserved 54-feature model

`feature_order.pkl` and `dataset_info.pkl` agree on these original model columns, in order:

```text
URL, URLLength, Domain, DomainLength, IsDomainIP, TLD, URLSimilarityIndex, CharContinuationRate, TLDLegitimateProb, URLCharProb, TLDLength, NoOfSubDomain, HasObfuscation, NoOfObfuscatedChar, ObfuscationRatio, NoOfLettersInURL, LetterRatioInURL, NoOfDegitsInURL, DegitRatioInURL, NoOfEqualsInURL, NoOfQMarkInURL, NoOfAmpersandInURL, NoOfOtherSpecialCharsInURL, SpacialCharRatioInURL, IsHTTPS, LineOfCode, LargestLineLength, HasTitle, Title, DomainTitleMatchScore, URLTitleMatchScore, HasFavicon, Robots, IsResponsive, NoOfURLRedirect, NoOfSelfRedirect, HasDescription, NoOfPopup, NoOfiFrame, HasExternalFormSubmit, HasSocialNet, HasSubmitButton, HasHiddenFields, HasPasswordField, Bank, Pay, Crypto, HasCopyrightInfo, NoOfImage, NoOfCSS, NoOfJS, NoOfSelfRef, NoOfEmptyRef, NoOfExternalRef
```

The fitted pipeline has a median `SimpleImputer` and `StandardScaler` for its 50 numeric inputs, and a most-frequent imputer plus `OneHotEncoder(handle_unknown="ignore")` for categorical `URL`, `Domain`, `TLD`, and `Title`. Its labels remain `0 = Phishing`, `1 = Legitimate`. Missing webpage values are never sent to this model or imputed for an incomplete request.

Dataset inspection corrected the live `Domain`, `DomainLength`, `TLD`, `TLDLength`, and `NoOfSubDomain` parsing to match the CSV convention: `Domain` is the full hostname, `TLD` is the final hostname label, and the subdomain count is the number of labels before the final two hostname labels. These corrections do not change the original model artifact. Other dataset-specific calculations that could not be confirmed remain missing in full webpage extraction.

## URL-only model schema

`URL_MODEL_FEATURE_ORDER` in `backend/url_features.py` defines the model inputs. All 34 values are computed from the same canonical URL helper in both training and inference. They include URL/host length, last-label TLD and length, IP/HTTPS/port flags, subdomain and path/query structure, ASCII letter/digit/special counts and ratios, selected punctuation/escape counts, and URL/host character entropy. `tld` is the only categorical input. No webpage text, domain identity, URL identity, filename, label-derived field, reputation lookup, similarity score, or precomputed PhiUSIIL feature is used.

The [UCI dataset record](https://archive.ics.uci.edu/dataset/967/phiusiil%2Bphishing%2Burl%2Bdataset) describes records as URLs and corresponding webpages and notes that some supplied features are derived, but does not provide their original formulas. The new model therefore uses its own explicitly defined URL-only features, recomputed from raw URL strings during training; it does not reuse ambiguous precomputed columns.

## Dataset audit and grouped evaluation

The trainer inspected the specified PhiUSIIL CSV:

- 235,795 rows and 56 columns: 54 features, `FILENAME`, and `label`.
- Labels: 100,945 phishing (`0`) and 134,850 legitimate (`1`). No missing values were found.
- No exact duplicate full records; 425 repeated raw URLs, with no conflicting labels for exact raw URLs.
- 34 URLs rejected by the live URL safety/normalization rules. After canonicalization, conflicting-label removal, and canonical URL deduplication, 234,862 unique URLs were used (100,013 phishing and 134,849 legitimate).
- 120 registrable-domain groups (1,374 source rows) contained both labels. Five-fold `StratifiedGroupKFold` evaluation grouped by Public Suffix List registrable domain; no registrable domain crossed a train/test fold.
- `FILENAME` numeric portion correlates with label (Pearson r = -0.604) and was excluded. The precomputed feature columns were also excluded; notably, `URLSimilarityIndex` averages 49.62 for phishing and 100 for legitimate, making it unsafe to treat as an independent raw URL input.
- All legitimate rows are HTTPS, compared with 48.5% of phishing rows. This is a major source-dataset bias and a strong shortcut for the fitted model. The filename-order final 20% contains only one class, so it cannot provide a valid two-class temporal/source holdout.

The trainer compares the saved URL-model XGBoost configuration (200 trees, depth 4, learning rate 0.06, `min_child_weight=3`, subsampling 0.85, L2=3) with a regularized candidate (300 trees, depth 3, learning rate 0.05, `min_child_weight=5`, subsampling 0.9, L2=5), using identical 5-fold domain-grouped splits. It selects higher macro-F1, then ROC-AUC on ties, retaining the baseline on a tie. The machine-readable report includes both sets of same-split metrics. The current saved configuration won, so the app retains the baseline XGBoost settings. On the shared registrable-domain folds, baseline versus candidate results were:

| Metric | Existing configuration | Candidate configuration |
|---|---:|---:|
| Accuracy | 0.9705 | 0.9681 |
| Macro precision | 0.9747 | 0.9729 |
| Macro recall | 0.9658 | 0.9630 |
| Macro F1 | 0.9696 | 0.9670 |
| ROC-AUC | 0.9956 | 0.9949 |

Selected-model out-of-fold figures come from the domain-grouped five-fold evaluation; probabilities treat legitimate (`1`) as the positive class for ROC-AUC:

| Metric | Grouped OOF result |
|---|---:|
| Accuracy | 0.9705 |
| Macro precision | 0.9747 |
| Macro recall | 0.9658 |
| Macro F1 | 0.9696 |
| ROC-AUC | 0.9956 |

Confusion matrix rows are true labels `[phishing, legitimate]`; columns are predicted labels `[phishing, legitimate]`:

```text
[[93462,  6551],
 [  380, 134469]]
```

Per class:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Phishing | 0.9960 | 0.9345 | 0.9642 | 100,013 |
| Legitimate | 0.9535 | 0.9972 | 0.9749 | 134,849 |

A no-HTTPS-feature grouped ablation scored 0.8929 accuracy, 0.9124 macro precision, 0.8771 macro recall, 0.8866 macro F1 and 0.9770 ROC-AUC; phishing recall fell to 0.7704. This reinforces that the full model leans heavily on the dataset's HTTPS pattern. A random stratified split scored 0.9904 accuracy with 2,786 overlapping registrable domains, so it is reported only as a leakage/overlap diagnostic, not the primary result.

These are dataset-held-out-domain results, not evidence of real-world effectiveness. Domain grouping prevents direct registrable-domain overlap, but it cannot remove collection-source, temporal, population, or label-acquisition bias. The model's output is raw and uncalibrated. Validate on an independent, contemporary, carefully labeled URL collection before relying on it operationally or making deployment-performance claims.

Reproduce training and evaluation (PowerShell):

```powershell
Set-Location 'C:\Users\deven\Documents\Codex\2026-10-12\files-pasted-by-the-user-build\outputs\phishguard'
$env:PYTHONPATH = (Resolve-Path .\backend).Path
python backend\train_url_model.py --dataset 'C:\Users\deven\OneDrive\Desktop\url_project\PhiUSIIL_Phishing_URL_Dataset.csv'
```

The script compares the existing URL-only configuration with a candidate, then updates only the URL-only artifact and evaluation JSON using the selected configuration. The original 54-feature model artifacts are not modified. It does not overwrite the original 54-feature model artifacts.

## Safe webpage retrieval

- Only HTTP(S) on standard ports; malformed URLs, URL credentials, local/internal/reserved/private/loopback/link-local addresses and metadata hosts are blocked.
- DNS resolution is bounded; every returned address is checked, and the connection is pinned to a checked IP.
- Every redirect target is revalidated; HTTPS-to-HTTP downgrade and invalid/excessive redirect chains are blocked.
- Environment proxies are disabled; connect/read/total time and decompressed response size are bounded.
- HTTP 401/403 are not bypassed; when the URL-only model is available, it can classify the submitted URL without reading the restricted page. Website JavaScript is never executed.
- Non-HTML, empty, and oversized content remain unavailable to the webpage extractor; URL-only mode may still run when the URL model is loaded, including for 401/403 without reading restricted page content.

## Local setup

Requirements verified in this workspace: Python 3.14.0 with the pinned model dependencies, Node 26.7.0 and npm 11.19.0. The model runtime pins scikit-learn 1.9.1 and XGBoost 3.4.1.

Backend, in the first PowerShell terminal:

```powershell
Set-Location 'C:\Users\deven\Documents\Codex\2026-10-12\files-pasted-by-the-user-build\outputs\phishguard'
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:PYTHONPATH = (Resolve-Path .\backend).Path
uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8000
```

Frontend, in the second terminal:

```powershell
Set-Location 'C:\Users\deven\Documents\Codex\2026-10-12\files-pasted-by-the-user-build\outputs\phishguard'
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
npm ci
npm run dev
```

The UI retains the single URL input and identifies the selected model and its version. URL-only result counts refer to the 34 URL model inputs; full-model counts refer to the original 54 inputs. Health reports separately whether the full and URL-only models loaded and returns HTTP 503 if neither is available.

## Production configuration and Render

The project has no `render.yaml` or other Render service definition. Configure two services manually from the same repository; the current FastAPI app does not serve the built frontend files.

1. **Backend web service** (repository root): build command `pip install -r requirements.txt`; start command `uvicorn --app-dir backend --host 0.0.0.0 --port $PORT --no-proxy-headers main:app`; health-check path `/health`. Configure `PHISHGUARD_CORS_ORIGINS` with the exact frontend origin after the static site is created. Create the backend first so its service origin can be used in the frontend build.
2. **Frontend static site** (repository root): build command `npm ci && npm run build`; publish directory `dist`. Set build-time `VITE_API_URL` to the backend origin. A production frontend with this value missing shows an explicit API-configuration error rather than falling back to a visitor's localhost.

| Variable | Service | Purpose and default |
|---|---|---|
| `VITE_API_URL` | Frontend build | Required in production; backend origin. Local development alone falls back to `http://127.0.0.1:8000`. |
| `PHISHGUARD_CORS_ORIGINS` | Backend | Comma-separated allowed frontend origins; defaults to the two localhost development origins. |
| `PHISHGUARD_MODEL_DIR` | Backend | Optional directory containing model artifacts; defaults to `backend/models`. |
| `PHISHGUARD_RATE_LIMIT_REQUESTS` | Backend | Requests per client key/window; default 30, valid range 1–10,000. Invalid values fall back to the default and are logged. |
| `PHISHGUARD_RATE_LIMIT_WINDOW_SECONDS` | Backend | Fixed-window length; default 60 seconds, valid range 1–86,400. Invalid values fall back to the default and are logged. |
| `PORT` | Backend | Set by Render and used by the start command. |

Set a compatible Python 3.14 release for the backend (the checked local interpreter is Python 3.14.0). The project does not pin a Python patch version. The installed Vite version declares Node support `^20.19.0 || >=22.12.0`; this workspace's build was checked with Node 26.7.0. The project does not pin Node, so select and pin a compatible version in the Render service configuration. The frontend dependencies are locked by `package-lock.json`; backend dependencies are declared in `requirements.txt` (some are version ranges).

The runtime needs `backend/models/feature_order.pkl` and `backend/models/dataset_info.pkl`, plus at least one valid classifier artifact: `phishguard_model.pkl` or `phishguard_url_model.joblib`. For this project, include both classifier artifacts and both schema files in the backend deploy. The evaluation JSON, confusion-matrix image, and legacy `model_comparison.csv` are not loaded by the API. The model files exist locally, but repository/remote inclusion was not verified.

The rate limiter is an in-memory, per-process fixed window. It uses the ASGI connection's `request.client.host` and ignores forwarded-IP headers. `--no-proxy-headers` prevents Uvicorn from deriving client identity from those headers. Behind Render's proxy, requests may therefore share the proxy's bucket; each worker or instance has its own counter. Treat this as a basic abuse deterrent, not a distributed or foolproof per-user limit. If the service is scaled or individual client identity is required, use a shared limiter and a verified trusted-proxy configuration.

`/health` returns HTTP 200 when the validated URL-only model required for current live analysis is loaded, and HTTP 503 otherwise. The preserved 54-feature artifact alone is not sufficient because the live extractor cannot produce its complete input schema. Configure `/health` as the Render readiness check; a missing URL-only model then prevents an unhealthy instance from receiving traffic. The API has no authentication. Before exposing it broadly, consider the intended audience and whether rate limits or additional abuse controls are sufficient.

The fetcher reads only bounded static HTML and never runs JavaScript or fetches page subresources. Dataset-specific features the extractor cannot determine remain missing. In the current code, at least 17 required 54-model fields are not assigned, so the original model will not receive live-extracted complete input; URL-only XGBoost is the usable live prediction path unless the extractor is later validated for every original field. The URL-only score is raw and uncalibrated.

## Verification

```powershell
Set-Location 'C:\Users\deven\Documents\Codex\2026-10-12\files-pasted-by-the-user-build\outputs\phishguard'
$env:PYTHONPATH = (Resolve-Path .\backend).Path
python -m unittest discover -s backend\tests -v
npm run build
```

The backend suite uses mocked network responses and tests XGBoost model schema/loading, full versus URL-only selection (including 401/403 fallback), explicit errors when no fallback model is available, correct labels and raw scores, deterministic repeated inference, DNS/timeouts, redirects, private/internal IP blocking, response limits, and static parsing. No separate frontend test runner is configured; production build runs TypeScript and Vite. A live browser-to-API end-to-end request could not be confirmed in this sandbox because it rejects local socket connections.
