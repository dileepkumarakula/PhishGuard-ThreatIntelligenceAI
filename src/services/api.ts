export type Health = { status: 'ok' | 'unavailable'; model_loaded: boolean; url_model_loaded?: boolean; analysis_ready?: boolean; model_type?: string; feature_count?: number; label_mapping?: Record<string,string>; message?: string }
export type Prediction = {
  analysis_status: 'complete'|'url_only'|'partial'|'unable_to_analyze'; analysis_mode: 'full_webpage'|'url_only'|'incomplete'; model_name: 'phiusiil_54_feature'|'phishguard_url_only'|null; model_version: string|null; url_model_features: Record<string,string|number|boolean>|null; prediction: 'phishing'|'legitimate'|null; label: 0|1|null; label_name: string|null; confidence: number|null;
  confidence_kind: 'raw_model_probability'|null; scanned_url: string; final_url: string;
  redirects: number; features_extracted: number; features_total: number;
  missing_features: string[]; extracted_features: Record<string,string|number|boolean|null>;
  warnings: string[]; imputation: string;
}
function apiBase() {
  const configured = import.meta.env.VITE_API_URL?.trim()
  if (configured) return configured.replace(/\/$/, '')
  if (import.meta.env.DEV) return 'http://127.0.0.1:8000'
  throw new Error('VITE_API_URL is required for a production frontend build.')
}
async function request<T>(path: string, init: RequestInit={}, timeoutMs=20_000, acceptUnavailable=false): Promise<T> {
  const controller = new AbortController(), timer = window.setTimeout(()=>controller.abort(),timeoutMs)
  try {
    const response = await fetch(`${apiBase()}${path}`, {...init, signal:controller.signal})
    const body = await response.json().catch(()=>null)
    if (!response.ok && !(acceptUnavailable && response.status===503)) throw new Error(typeof body?.detail==='string'?body.detail:`Request failed (${response.status}).`)
    return body as T
  } catch (error) {
    if (error instanceof DOMException && error.name==='AbortError') throw new Error('The scan timed out. Try again or use a smaller, reachable page.')
    if (error instanceof TypeError) throw new Error('Could not reach the PhishGuard API. Confirm the backend is running and VITE_API_URL is correct.')
    throw error
  } finally { window.clearTimeout(timer) }
}
export function checkHealth() { return request<Health>('/health',{},5_000,true) }
export function predictUrl(url:string) { return request<Prediction>('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({url})}) }
