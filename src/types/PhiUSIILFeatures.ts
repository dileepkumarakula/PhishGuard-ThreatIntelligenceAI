export interface PhiUSIILFeatures {
  URL:string; URLLength:number; Domain:string; DomainLength:number; IsDomainIP:number; TLD:string; URLSimilarityIndex:number; CharContinuationRate:number; TLDLegitimateProb:number; URLCharProb:number; TLDLength:number; NoOfSubDomain:number; HasObfuscation:number; NoOfObfuscatedChar:number; ObfuscationRatio:number; NoOfLettersInURL:number; LetterRatioInURL:number; NoOfDegitsInURL:number; DegitRatioInURL:number; NoOfEqualsInURL:number; NoOfQMarkInURL:number; NoOfAmpersandInURL:number; NoOfOtherSpecialCharsInURL:number; SpacialCharRatioInURL:number; IsHTTPS:number;
  LineOfCode:number; LargestLineLength:number; HasTitle:number; Title:string; DomainTitleMatchScore:number; URLTitleMatchScore:number; HasFavicon:number; Robots:number; IsResponsive:number; NoOfURLRedirect:number; NoOfSelfRedirect:number; HasDescription:number; NoOfPopup:number; NoOfiFrame:number; HasExternalFormSubmit:number; HasSocialNet:number; HasSubmitButton:number; HasHiddenFields:number; HasPasswordField:number;
  Bank:number; Pay:number; Crypto:number; HasCopyrightInfo:number; NoOfImage:number; NoOfCSS:number; NoOfJS:number; NoOfSelfRef:number; NoOfEmptyRef:number; NoOfExternalRef:number;
}
export type FeatureKey = keyof PhiUSIILFeatures;
export type FeatureSource = 'AUTO'|'MANUAL'|'BACKEND';
export type FeatureValues = Record<FeatureKey, string | number>;
