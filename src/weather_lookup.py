"""Weather API helper utilities.

This module follows the WeatherAPI.com Swagger definition published at:
https://app.swaggerhub.com/apis-docs/WeatherAPI.com/WeatherAPI/1.0.2
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

import httpx

DEFAULT_WEATHERAPI_BASE_URL = "https://api.weatherapi.com/v1"


class WeatherLookupError(RuntimeError):
    """Raised when WeatherAPI.com lookups fail or return invalid responses."""


@dataclass(slots=True)
class WeatherAPIConfig:
    """Configuration container for WeatherAPI.com requests."""

    api_key: str
    base_url: str = DEFAULT_WEATHERAPI_BASE_URL
    timeout_seconds: float = 10.0
    default_language: Optional[str] = None

    @classmethod
    def from_env(cls) -> "WeatherAPIConfig":
        """Create a configuration object from environment variables or .env values."""
        api_key = (
            os.getenv("WEATHER_API_KEY")
            or os.getenv("WEATHERAPI_KEY")
            or os.getenv("WEATHERAPI_API_KEY")
        )
        if not api_key:
            raise WeatherLookupError(
                "WeatherAPI configuration requires the WEATHER_API_KEY environment variable."
            )

        base_url = os.getenv("WEATHER_API_BASE_URL") or DEFAULT_WEATHERAPI_BASE_URL
        language = os.getenv("WEATHER_API_LANG") or None
        timeout_value = os.getenv("WEATHER_API_TIMEOUT")
        timeout_seconds = 10.0
        if timeout_value:
            try:
                timeout_seconds = float(timeout_value)
            except ValueError as exc:
                raise WeatherLookupError(
                    "WEATHER_API_TIMEOUT must be a number representing seconds."
                ) from exc

        return cls(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            timeout_seconds=timeout_seconds,
            default_language=language,
        )


@dataclass(slots=True)
class WeatherObservation:
    """Key current weather fields extracted from the `/current.json` response."""

    location_name: Optional[str]
    region: Optional[str]
    country: Optional[str]
    local_time: Optional[str]
    condition_text: Optional[str]
    temperature_c: Optional[float]
    temperature_f: Optional[float]
    feelslike_c: Optional[float]
    feelslike_f: Optional[float]
    humidity: Optional[int]
    wind_kph: Optional[float]
    wind_mph: Optional[float]
    raw: dict[str, Any]

    def summary(self) -> str:
        """Return a concise summary string for conversational use."""
        location = ", ".join(
            value for value in (self.location_name, self.region, self.country) if value
        )
        condition = self.condition_text or "Unknown conditions"
        temp_c = (
            f"{self.temperature_c:.1f}°C"
            if isinstance(self.temperature_c, (int, float))
            else "?"
        )
        temp_f = (
            f"{self.temperature_f:.1f}°F"
            if isinstance(self.temperature_f, (int, float))
            else "?"
        )
        return f"{location or 'Unknown location'}: {condition}, {temp_c}/{temp_f}"


async def lookup_current_weather(
    location_query: str,
    *,
    config: Optional[WeatherAPIConfig] = None,
    lang: Optional[str] = None,
    client: Optional[httpx.AsyncClient] = None,
) -> WeatherObservation:
    """Fetch the current weather for the supplied location string.

    The `/current.json` endpoint defined in the WeatherAPI.com Swagger specification accepts:
      * `key` – API key (required)
      * `q` – location query such as city name, ZIP/postal code, IP address, or `lat,long` (required)
      * `lang` – language code for localized condition text (optional)

    Args:
        location_query: The value passed to the WeatherAPI `q` query parameter.
        config: Optional explicit configuration. When omitted the environment is consulted.
        lang: Optional language override for this lookup.
        client: Optional shared HTTP client. When absent, a short-lived client is used.

    Raises:
        WeatherLookupError: When the API request fails or returns an unexpected payload.
        ValueError: When `location_query` is empty or whitespace.
    """
    if not location_query or not location_query.strip():
        raise ValueError("location_query must be a non-empty string.")

    resolved_config = config or WeatherAPIConfig.from_env()
    params = {
        "key": resolved_config.api_key,
        "q": location_query.strip(),
    }
    resolved_lang = lang or resolved_config.default_language
    if resolved_lang:
        params["lang"] = resolved_lang

    url = f"{resolved_config.base_url.rstrip('/')}/current.json"
    timeout = httpx.Timeout(resolved_config.timeout_seconds, connect=5.0)

    try:
        if client is None:
            async with httpx.AsyncClient(timeout=timeout) as session:
                response = await session.get(url, params=params)
        else:
            response = await client.get(url, params=params, timeout=timeout)
    except httpx.TimeoutException as exc:
        raise WeatherLookupError("WeatherAPI lookup timed out.") from exc
    except httpx.HTTPError as exc:
        raise WeatherLookupError(f"WeatherAPI lookup failed: {exc}") from exc

    if response.status_code >= 400:
        error_message = response.text.strip() or f"HTTP {response.status_code}"
        try:
            payload = response.json()
        except ValueError:
            payload = None
        if isinstance(payload, dict):
            error_info = payload.get("error") or {}
            code = error_info.get("code")
            message = error_info.get("message")
            if message:
                error_message = (
                    f"{message} (code {code})"
                    if code is not None
                    else message
                )
        raise WeatherLookupError(f"WeatherAPI returned {response.status_code}: {error_message}")

    try:
        data = response.json()
    except ValueError as exc:
        raise WeatherLookupError("WeatherAPI returned a non-JSON response.") from exc

    if not isinstance(data, dict):
        raise WeatherLookupError("WeatherAPI response was not an object.")

    location_info = data.get("location")
    current_info = data.get("current")
    if not isinstance(location_info, dict) or not isinstance(current_info, dict):
        raise WeatherLookupError("WeatherAPI response missing 'location' or 'current' sections.")

    condition = current_info.get("condition") if isinstance(current_info.get("condition"), dict) else {}

    observation = WeatherObservation(
        location_name=_safe_str(location_info.get("name")),
        region=_safe_str(location_info.get("region")),
        country=_safe_str(location_info.get("country")),
        local_time=_safe_str(location_info.get("localtime")),
        condition_text=_safe_str(condition.get("text")),
        temperature_c=_safe_float(current_info.get("temp_c")),
        temperature_f=_safe_float(current_info.get("temp_f")),
        feelslike_c=_safe_float(current_info.get("feelslike_c")),
        feelslike_f=_safe_float(current_info.get("feelslike_f")),
        humidity=_safe_int(current_info.get("humidity")),
        wind_kph=_safe_float(current_info.get("wind_kph")),
        wind_mph=_safe_float(current_info.get("wind_mph")),
        raw=data,
    )
    return observation


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_str(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return value
    return value if isinstance(value, str) else None

