import type { FeatureKey, FeatureValues } from '../types/PhiUSIILFeatures'
const fields: FeatureKey[] = ['URL','URLLength','Domain','DomainLength','IsDomainIP','TLD','URLSimilarityIndex','CharContinuationRate','TLDLegitimateProb','URLCharProb','TLDLength','NoOfSubDomain','HasObfuscation','NoOfObfuscatedChar','ObfuscationRatio','NoOfLettersInURL','LetterRatioInURL','NoOfDegitsInURL','DegitRatioInURL','NoOfEqualsInURL','NoOfQMarkInURL','NoOfAmpersandInURL','NoOfOtherSpecialCharsInURL','SpacialCharRatioInURL','IsHTTPS','LineOfCode','LargestLineLength','HasTitle','Title','DomainTitleMatchScore','URLTitleMatchScore','HasFavicon','Robots','IsResponsive','NoOfURLRedirect','NoOfSelfRedirect','HasDescription','NoOfPopup','NoOfiFrame','HasExternalFormSubmit','HasSocialNet','HasSubmitButton','HasHiddenFields','HasPasswordField','Bank','Pay','Crypto','HasCopyrightInfo','NoOfImage','NoOfCSS','NoOfJS','NoOfSelfRef','NoOfEmptyRef','NoOfExternalRef']
export const FEATURE_KEYS = fields
export const BINARY_KEYS: FeatureKey[] = ['IsDomainIP','HasObfuscation','IsHTTPS','HasTitle','HasFavicon','Robots','IsResponsive','HasDescription','HasExternalFormSubmit','HasSocialNet','HasSubmitButton','HasHiddenFields','HasPasswordField','Bank','Pay','Crypto','HasCopyrightInfo']
export const AUTO_KEYS: FeatureKey[] = ['URL','URLLength','Domain','DomainLength','IsDomainIP','TLD','TLDLength','NoOfSubDomain','NoOfLettersInURL','LetterRatioInURL','NoOfDegitsInURL','DegitRatioInURL','NoOfEqualsInURL','NoOfQMarkInURL','NoOfAmpersandInURL','NoOfOtherSpecialCharsInURL','SpacialCharRatioInURL','IsHTTPS']
export const NUMERIC_KEYS = FEATURE_KEYS.filter(k => k !== 'URL' && k !== 'Domain' && k !== 'TLD' && k !== 'Title')
export const defaults = Object.fromEntries(fields.map(k => [k, ['URL','Domain','TLD','Title'].includes(k) ? '' : ''])) as FeatureValues
export function extractUrl(raw: string): Partial<FeatureValues> {
 if (!raw.trim()) return {}
 try {
  const normalized = /^[a-z][a-z\d+.-]*:\/\//i.test(raw) ? raw : `https://${raw}`
  const u = new URL(normalized), host = u.hostname.toLowerCase(), labels = host.split('.'), domain = labels.length > 2 ? labels.slice(-2).join('.') : host
  const text = raw, letters = (text.match(/[a-z]/gi)||[]).length, digits = (text.match(/\d/g)||[]).length, specials = (text.match(/[^a-z\d]/gi)||[]).length
  const ip = /^\d{1,3}(\.\d{1,3}){3}$/.test(host), tld = ip ? '' : `.${labels.at(-1) || ''}`
  const obfuscated = (text.match(/%[\da-f]{2}/gi)||[]).length
  return {URL: raw, URLLength:text.length, Domain:host, DomainLength:domain.length, TLD:tld, TLDLength:tld.replace('.','').length, NoOfSubDomain:Math.max(0,labels.length-(ip?4:2)), IsDomainIP:Number(ip), IsHTTPS:Number(u.protocol==='https:'), NoOfLettersInURL:letters, LetterRatioInURL:letters/(text.length||1), NoOfDegitsInURL:digits, DegitRatioInURL:digits/(text.length||1), NoOfEqualsInURL:(text.match(/=/g)||[]).length, NoOfQMarkInURL:(text.match(/\?/g)||[]).length, NoOfAmpersandInURL:(text.match(/&/g)||[]).length, NoOfOtherSpecialCharsInURL:(text.match(/[!#$%^*()_+\-=[\]{};':"\\|,.<>/~`]/g)||[]).length, SpacialCharRatioInURL:specials/(text.length||1), NoOfObfuscatedChar:obfuscated, HasObfuscation:Number(obfuscated>0), ObfuscationRatio:obfuscated/(text.length||1)}
 } catch { return { URL: raw } }
}
