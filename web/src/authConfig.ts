import type { Configuration, PopupRequest } from "@azure/msal-browser";

const fallbackClientId = "00000000-0000-0000-0000-000000000000";
const fallbackAuthority = "https://login.microsoftonline.com/common";

const rawClientId = (import.meta.env.VITE_AAD_CLIENT_ID ?? "").trim();
const rawTenantId = (import.meta.env.VITE_AAD_TENANT_ID ?? "").trim();
const rawAuthority = (import.meta.env.VITE_AAD_AUTHORITY ?? "").trim();
const rawRedirectUri = (import.meta.env.VITE_AAD_REDIRECT_URI ?? "").trim();
const rawDisable = (import.meta.env.VITE_AAD_AUTH_DISABLED ?? "").trim().toLowerCase();
const rawScope = (import.meta.env.VITE_AAD_API_SCOPE ?? "").trim();
const rawAllowedTenants = (import.meta.env.VITE_AAD_ALLOWED_TENANTS ?? "").trim();

const resolvedAuthority =
  rawAuthority || (rawTenantId ? `https://login.microsoftonline.com/${rawTenantId}` : fallbackAuthority);
const resolvedClientId = rawClientId || fallbackClientId;

const disabledViaFlag = rawDisable === "1" || rawDisable === "true" || rawDisable === "yes";
export const azureAuthEnabled = Boolean(rawClientId && (rawTenantId || rawAuthority) && !disabledViaFlag);

const parseList = (value: string): string[] =>
  value
    .split(/[\s,]+/)
    .map((scope) => scope.trim())
    .filter(Boolean);

const computedScopes = azureAuthEnabled ? parseList(rawScope) : [];
if (azureAuthEnabled && computedScopes.length === 0 && rawClientId) {
  computedScopes.push(`api://${rawClientId}/.default`);
}

const redirectUri = rawRedirectUri || window.location.origin;

export const msalConfig: Configuration = {
  auth: {
    clientId: resolvedClientId,
    authority: resolvedAuthority,
    redirectUri,
  },
  cache: {
    cacheLocation: "localStorage",
    storeAuthStateInCookie: false,
  },
};

export const loginRequest: PopupRequest = {
  scopes: [...computedScopes],
};

export const apiScopes: readonly string[] = computedScopes;

const computedAllowedTenants = parseList(rawAllowedTenants).map((tenant) => tenant.toLowerCase());

export const allowedTenantIds: readonly string[] = computedAllowedTenants;
export const tenantRestrictionEnabled = computedAllowedTenants.length > 0;
