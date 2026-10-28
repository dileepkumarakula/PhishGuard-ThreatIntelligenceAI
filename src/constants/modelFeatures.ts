import type { FeatureKey } from '../types/PhiUSIILFeatures'
// Keep this list aligned with the columns selected for training, including text columns when the pipeline uses them.
export const MODEL_FEATURE_ORDER: FeatureKey[] = [
  'URL','URLLength','Domain','DomainLength','IsDomainIP','TLD','URLSimilarityIndex','CharContinuationRate','TLDLegitimateProb','URLCharProb','TLDLength','NoOfSubDomain','HasObfuscation','NoOfObfuscatedChar','ObfuscationRatio','NoOfLettersInURL','LetterRatioInURL','NoOfDegitsInURL','DegitRatioInURL','NoOfEqualsInURL','NoOfQMarkInURL','NoOfAmpersandInURL','NoOfOtherSpecialCharsInURL','SpacialCharRatioInURL','IsHTTPS',
  'LineOfCode','LargestLineLength','HasTitle','Title','DomainTitleMatchScore','URLTitleMatchScore','HasFavicon','Robots','IsResponsive','NoOfURLRedirect','NoOfSelfRedirect','HasDescription','NoOfPopup','NoOfiFrame','HasExternalFormSubmit','HasSocialNet','HasSubmitButton','HasHiddenFields','HasPasswordField',
  'Bank','Pay','Crypto','HasCopyrightInfo','NoOfImage','NoOfCSS','NoOfJS','NoOfSelfRef','NoOfEmptyRef','NoOfExternalRef'
]
export const FEATURE_ORDER_NOTE = 'Configure MODEL_FEATURE_ORDER to match the column order used by your trained model.'
