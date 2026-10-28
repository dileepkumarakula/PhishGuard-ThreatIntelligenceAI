import type { FeatureValues } from '../types/PhiUSIILFeatures'
import { BINARY_KEYS, FEATURE_KEYS, NUMERIC_KEYS } from '../constants/features'
export function validate(values: FeatureValues) {
 const errors: Record<string,string> = {}
 if (typeof values.URL !== 'string' || !values.URL.trim()) errors.URL = 'URL is required.'
 else { try { new URL(/^[a-z][a-z\d+.-]*:\/\//i.test(String(values.URL)) ? String(values.URL) : `https://${values.URL}`) } catch { errors.URL = 'Enter a valid URL or domain.' } }
 for (const k of FEATURE_KEYS) {
  if (k === 'URL') continue
  const v = values[k]
  if (v === '' || v === undefined || v === null) { errors[k] = 'This field is required.'; continue }
  if (BINARY_KEYS.includes(k)) { if (![0,1,'0','1'].includes(v as never)) errors[k] = 'Choose Yes or No.'; continue }
  if (new Set<string>(NUMERIC_KEYS).has(k)) { const n=Number(v); if (!Number.isFinite(n)) errors[k]='Enter a valid number.'; else if (n<0) errors[k]='Value must be zero or greater.'; else if (['LetterRatioInURL','DegitRatioInURL','SpacialCharRatioInURL','ObfuscationRatio','TLDLegitimateProb','URLCharProb','URLSimilarityIndex','CharContinuationRate','DomainTitleMatchScore','URLTitleMatchScore'].includes(k) && n>1 && !['URLSimilarityIndex','DomainTitleMatchScore','URLTitleMatchScore'].includes(k)) errors[k]='Enter a value between 0 and 1.' }
 }
 return errors
}
