"""
Azure AD authentication helpers for the Zara Voice Service backend.

This module provides:
  * Environment-driven configuration for Azure AD (Entra ID) validation.
  * Bearer token validation using the JSON Web Key Set published by Azure AD.
  * FastAPI dependencies/utilities for HTTP and WebSocket endpoints.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable, Optional

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import PyJWKClient
from jwt.exceptions import PyJWTError
from starlette.websockets import WebSocket


logger = logging.getLogger("zara.auth")


class AuthenticationError(Exception):
    """Raised when bearer token validation fails."""


def _parse_env_set(value: Optional[str]) -> frozenset[str]:
    if not value:
        return frozenset()
    separators = {",", " ", ";", "\n"}
    tokens: list[str] = []
    current = []
    for char in value:
        if char in separators:
            token = "".join(current).strip()
            if token:
                tokens.append(token)
            current = []
        else:
            current.append(char)
    last_token = "".join(current).strip()
    if last_token:
        tokens.append(last_token)
    return frozenset(tokens)


@dataclass(frozen=True, slots=True)
class AzureADSettings:
    enabled: bool
    audience: Optional[str] = None
    audiences: tuple[str, ...] = ()
    tenant_id: Optional[str] = None
    client_id: Optional[str] = None
    authority: Optional[str] = None
    issuer: Optional[str] = None
    jwks_uri: Optional[str] = None
    allowed_client_ids: frozenset[str] = frozenset()
    required_scopes: frozenset[str] = frozenset()
    required_roles: frozenset[str] = frozenset()
    allowed_tenants: frozenset[str] = frozenset()
    enforce_known_tenants: bool = False

    def __post_init__(self) -> None:
        if self.enabled and not (self.client_id and self.audience):
            raise ValueError("Azure AD auth is enabled but client/audience settings are incomplete.")


@lru_cache
def get_azure_ad_settings() -> AzureADSettings:
    """
    Load the Azure AD configuration from environment variables.

    Required:
        AZURE_AD_TENANT_ID (or set AZURE_AD_AUTHORITY to a multi-tenant endpoint)
        AZURE_AD_CLIENT_ID
    Optional:
        AZURE_AD_API_AUDIENCE (defaults to client id)
        AZURE_AD_AUTHORITY (defaults to https://login.microsoftonline.com/{tenant})
        AZURE_AD_ISSUER (defaults to {authority}/v2.0)
        AZURE_AD_JWKS_URI (defaults to {authority}/discovery/v2.0/keys)
        AZURE_AD_ALLOWED_CLIENT_IDS (comma/space separated list)
        AZURE_AD_REQUIRED_SCOPES (space/comma separated list)
        AZURE_AD_REQUIRED_ROLES (space/comma separated list)
        AZURE_AD_ALLOWED_TENANT_IDS (comma/space separated list of accepted tenant IDs for multi-tenant apps)
        AZURE_AD_AUTH_DISABLED (when "1"/"true"/"yes", disables auth)
    """
    disabled_value = os.getenv("AZURE_AD_AUTH_DISABLED", "").lower()
    if disabled_value in {"1", "true", "yes"}:
        logger.warning("Azure AD authentication disabled via AZURE_AD_AUTH_DISABLED.")
        return AzureADSettings(enabled=False)

    client_id = os.getenv("AZURE_AD_CLIENT_ID", "").strip()
    if not client_id:
        logger.info("Azure AD authentication disabled: client id not configured.")
        return AzureADSettings(enabled=False)

    tenant_id = os.getenv("AZURE_AD_TENANT_ID", "").strip()
    tenant_hint = tenant_id.lower()
    multi_tenant_authorities = {"", "common", "organizations", "consumers"}

    audience_raw = os.getenv("AZURE_AD_API_AUDIENCE", "").strip()
    audience_values: set[str] = set()
    if audience_raw:
        audience_values.add(audience_raw)
        if audience_raw.endswith("/.default"):
            audience_values.add(audience_raw[:-9])
    audience_values.add(client_id)
    additional_audiences = _parse_env_set(os.getenv("AZURE_AD_ADDITIONAL_AUDIENCES"))
    audience_values.update(additional_audiences)
    audience_values = {value for value in audience_values if value}
    if not audience_values:
        audience_values.add(client_id)
    primary_audience = audience_raw or client_id

    authority = os.getenv("AZURE_AD_AUTHORITY", "").strip()
    if not authority:
        authority_tenant = tenant_id if tenant_id else "common"
        authority = f"https://login.microsoftonline.com/{authority_tenant}"
    authority = authority.rstrip("/")
    authority_segment = authority.rsplit("/", 1)[-1].lower()
    is_multi_authority = authority_segment in {"common", "organizations", "consumers"}

    issuer_env = os.getenv("AZURE_AD_ISSUER", "").strip()
    if issuer_env:
        issuer = issuer_env
    elif tenant_id and tenant_hint not in multi_tenant_authorities and not is_multi_authority:
        issuer = f"{authority}/v2.0"
    else:
        issuer = None

    jwks_uri = os.getenv("AZURE_AD_JWKS_URI", "").strip() or f"{authority}/discovery/v2.0/keys"

    allowed_client_ids_set = {value.lower() for value in _parse_env_set(os.getenv("AZURE_AD_ALLOWED_CLIENT_IDS"))}
    if client_id:
        allowed_client_ids_set.add(client_id.lower())
    allowed_client_ids = frozenset(allowed_client_ids_set)
    required_scopes = _parse_env_set(os.getenv("AZURE_AD_REQUIRED_SCOPES"))
    required_roles = _parse_env_set(os.getenv("AZURE_AD_REQUIRED_ROLES"))
    allowed_tenants = frozenset(tenant.lower() for tenant in _parse_env_set(os.getenv("AZURE_AD_ALLOWED_TENANT_IDS")))
    enforce_known_tenants = bool(allowed_tenants)
    if not allowed_tenants and tenant_id and tenant_hint not in multi_tenant_authorities:
        allowed_tenants = frozenset({tenant_id.lower()})
        enforce_known_tenants = True

    logger.info(
        "Azure AD authentication enabled; audience=%s authority=%s tenants=%s",
        primary_audience,
        authority,
        ",".join(sorted(allowed_tenants)) if allowed_tenants else "(any)",
    )

    return AzureADSettings(
        enabled=True,
        audience=primary_audience,
        audiences=tuple(sorted(audience_values)),
        tenant_id=tenant_id,
        client_id=client_id,
        authority=authority,
        issuer=issuer,
        jwks_uri=jwks_uri,
        allowed_client_ids=allowed_client_ids,
        required_scopes=required_scopes,
        required_roles=required_roles,
        allowed_tenants=allowed_tenants,
        enforce_known_tenants=enforce_known_tenants,
    )


@lru_cache
def _get_jwk_client() -> PyJWKClient:
    settings = get_azure_ad_settings()
    if not settings.enabled or not settings.jwks_uri:
        raise RuntimeError("JWK client requested but Azure AD auth is not configured.")
    return PyJWKClient(settings.jwks_uri)


def _validate_set_membership(
    actual: Iterable[str],
    required: frozenset[str],
    claim_name: str,
) -> None:
    if not required:
        return
    actual_set = {value.strip() for value in actual if value.strip()}
    missing = [item for item in required if item not in actual_set]
    if missing:
        raise AuthenticationError(f"Token missing required {claim_name}: {', '.join(missing)}")


def _extract_token_scopes(payload: dict[str, Any]) -> set[str]:
    scopes_raw = payload.get("scp")
    if isinstance(scopes_raw, str):
        return {scope for scope in scopes_raw.split() if scope}
    if isinstance(scopes_raw, list):
        return {scope for scope in scopes_raw if isinstance(scope, str)}
    return set()


def _extract_token_roles(payload: dict[str, Any]) -> set[str]:
    roles_raw = payload.get("roles")
    if isinstance(roles_raw, str):
        return {role for role in roles_raw.split() if role}
    if isinstance(roles_raw, list):
        return {role for role in roles_raw if isinstance(role, str)}
    return set()


def validate_bearer_token(token: str) -> dict[str, Any]:
    settings = get_azure_ad_settings()
    if not settings.enabled:
        raise AuthenticationError("Azure AD authentication is not enabled.")

    try:
        signing_key = _get_jwk_client().get_signing_key_from_jwt(token)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to resolve signing key for token: %s", exc)
        raise AuthenticationError("Unable to resolve token signing key.") from exc

    decode_kwargs: dict[str, object] = {
        "algorithms": ["RS256"],
        "options": {
            "verify_signature": True,
            "verify_aud": True,
            "verify_iss": False,
        },
    }
    if settings.issuer:
        decode_kwargs["issuer"] = settings.issuer
        decode_kwargs["options"]["verify_iss"] = True
    audiences = list(settings.audiences or ())
    if audiences:
        decode_kwargs["audience"] = audiences if len(audiences) > 1 else audiences[0]
    elif settings.audience:
        decode_kwargs["audience"] = settings.audience

    try:
        payload = jwt.decode(
            token,
            signing_key.key,
            **decode_kwargs,
        )
    except PyJWTError as exc:
        logger.debug("JWT decode error: %s", exc)
        raise AuthenticationError(f"Invalid bearer token: {exc}") from exc

    issuer_claim = str(payload.get("iss") or "")
    issuer_claim_lower = issuer_claim.lower()
    tenant_claim = str(payload.get("tid") or payload.get("tenantId") or "").lower()
    if settings.enforce_known_tenants:
        logger.debug(
            "Checking tenant/issuer: tenant_claim=%s allowed=%s issuer_claim=%s",
            tenant_claim,
            settings.allowed_tenants,
            issuer_claim_lower,
        )
        if not tenant_claim or tenant_claim not in settings.allowed_tenants:
            raise AuthenticationError("Caller tenant is not allowed to access this API.")
        expected_issuers: set[str] = set()
        for tenant in settings.allowed_tenants:
            expected_issuers.add(f"https://login.microsoftonline.com/{tenant}/v2.0")
            expected_issuers.add(f"https://login.microsoftonline.com/{tenant}/")
            expected_issuers.add(f"https://sts.windows.net/{tenant}/")
        if settings.issuer:
            expected_issuers.add(settings.issuer.lower())
        expected_issuers = {value.lower() for value in expected_issuers}
        logger.debug("Expected issuers: %s", expected_issuers)
        if issuer_claim and issuer_claim_lower not in expected_issuers:
            raise AuthenticationError(
                f"Token issuer '{issuer_claim_lower}' is not recognized (expected one of {sorted(expected_issuers)})."
            )
    else:
        if settings.issuer:
            if issuer_claim != settings.issuer:
                raise AuthenticationError("Token issuer mismatch.")
        elif issuer_claim and not (
            issuer_claim_lower.startswith("https://login.microsoftonline.com/")
            or issuer_claim_lower.startswith("https://sts.windows.net/")
            or issuer_claim_lower.startswith("https://sts.windows-ppe.net/")
        ):
            raise AuthenticationError(
                f"Token issuer '{issuer_claim_lower}' is not trusted for multi-tenant validation."
            )

    if settings.allowed_client_ids:
        client_claim_raw = (payload.get("azp") or payload.get("appid") or payload.get("client_id") or "").lower()
        if not client_claim_raw or client_claim_raw not in settings.allowed_client_ids:
            raise AuthenticationError(
                f"Client application '{client_claim_raw or 'unknown'}' is not allowed to call this API."
            )

    scopes = _extract_token_scopes(payload)
    _validate_set_membership(scopes, settings.required_scopes, "scopes")

    roles = _extract_token_roles(payload)
    _validate_set_membership(roles, settings.required_roles, "roles")

    logger.debug(
        "Validated bearer token: aud=%s iss=%s tenant=%s client=%s scopes=%s",
        payload.get("aud"),
        issuer_claim,
        tenant_claim,
        payload.get("azp") or payload.get("appid"),
        scopes,
    )

    return payload


http_bearer = HTTPBearer(auto_error=False)


def get_current_user_claims(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(http_bearer),
) -> Optional[dict[str, Any]]:
    settings = get_azure_ad_settings()
    if not settings.enabled:
        return None

    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    try:
        return validate_bearer_token(token)
    except AuthenticationError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


async def require_websocket_auth(websocket: WebSocket) -> Optional[dict[str, Any]]:
    settings = get_azure_ad_settings()
    if not settings.enabled:
        return None

    token = None
    auth_header = websocket.headers.get("authorization")
    if auth_header and auth_header.lower().startswith("bearer "):
        token = auth_header.split(" ", 1)[1].strip()

    if not token:
        token = websocket.query_params.get("access_token")

    if not token:
        raise AuthenticationError("Missing websocket access token.")

    return validate_bearer_token(token)
