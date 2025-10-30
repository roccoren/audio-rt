#!/usr/bin/env bash

# Deploys the Zara Voice backend and frontend to Azure Container Apps using the
# env values defined in .env and web/.env. The script builds/pushes Docker
# images to an Azure Container Registry and wires all environment variables
# (secrets are stored as Container App secrets automatically).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BACKEND_ENV_FILE="${REPO_ROOT}/.env"
FRONTEND_ENV_FILE="${REPO_ROOT}/web/.env"
DEFAULT_BACKEND_IMAGE_NAME="zara-voice-api"
DEFAULT_FRONTEND_IMAGE_NAME="zara-voice-web"
DEFAULT_BACKEND_TAG="latest"
DEFAULT_FRONTEND_TAG="latest"
DEFAULT_BACKEND_PORT=8000
DEFAULT_FRONTEND_PORT=80

usage() {
  cat <<'EOF'
Deploy the Zara Voice web + API stack to Azure Container Apps.

Required arguments:
  --resource-group <name>       Azure resource group for the deployment
  --location <azure-region>     Azure region for the resource group/environment
  --environment <name>          Container Apps managed environment name
  --acr-name <name>             Existing Azure Container Registry name (no FQDN)

Optional arguments:
  --subscription <id-or-name>   Azure subscription to target
  --workspace-name <name>       Existing/new Log Analytics workspace name
  --registry-username <user>    Container registry username (auto-discovered when omitted)
  --registry-password <pass>    Container registry password (auto-discovered when omitted)
  --backend-app-name <name>     Container App name for the API backend (default: zara-voice-api)
  --frontend-app-name <name>    Container App name for the static web app (default: zara-voice-web)
  --backend-image <tag>         Full image reference for the backend (overrides name/tag defaults)
  --frontend-image <tag>        Full image reference for the frontend (overrides name/tag defaults)
  --backend-image-name <name>   Image repository name within ACR (default: zara-voice-api)
  --frontend-image-name <name>  Image repository name within ACR (default: zara-voice-web)
  --backend-tag <tag>           Image tag for the backend build (default: latest)
  --frontend-tag <tag>          Image tag for the frontend build (default: latest)
  --backend-port <port>         Ingress target port for the API (default: 8000)
  --frontend-port <port>        Ingress target port for the web app (default: 80)
  --skip-build                  Skip docker build/push (assumes images already exist)
  --help, -h                    Show help

Example:
  ./infra/containerapps/deploy.sh \
    --resource-group voice-demo-rg \
    --location eastus \
    --environment voice-demo-env \
    --acr-name myacr \
    --backend-app-name zara-voice-api \
    --frontend-app-name zara-voice-web
EOF
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: '$1' command is required but was not found in PATH." >&2
    exit 1
  fi
}

# Basic env file parsing that respects quoting and inline comments.
parse_env_file() {
  local file_path="$1"
  python3 - "$file_path" <<'PY'
import sys
import re

path = sys.argv[1]
pattern = re.compile(r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$')

def parse_value(raw):
    value = raw.rstrip()
    if not value:
        return ""
    if value[0] in ('"', "'"):
        quote = value[0]
        value = value[1:]
        buf = []
        i = 0
        while i < len(value):
            ch = value[i]
            if ch == '\\' and i + 1 < len(value):
                buf.append(value[i + 1])
                i += 2
                continue
            if ch == quote:
                break
            buf.append(ch)
            i += 1
        return "".join(buf)
    # For unquoted values treat a # preceded by whitespace as a comment.
    comment_idx = None
    for idx, ch in enumerate(value):
        if ch == '#' and (idx == 0 or value[idx - 1].isspace()):
            comment_idx = idx
            break
    if comment_idx is not None:
        value = value[:comment_idx]
    return value.strip()

with open(path, "r", encoding="utf-8") as f:
    for raw_line in f:
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        match = pattern.match(raw_line)
        if not match:
            continue
        key, raw_value = match.group(1), match.group(2)
        value = parse_value(raw_value)
        sys.stdout.write(f"{key}\t{value}\n")
PY
}

is_secret_key() {
  case "$1" in
    *_KEY|*_SECRET|*CLIENT_SECRET|*_TOKEN|*_PASSWORD|TOKEN|PASSWORD)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

sanitize_secret_name() {
  local raw="$1"
  local name
  name="$(printf '%s' "${raw}" | tr '[:upper:]' '[:lower:]')"
  name="${name//_/-}"
  # Replace any remaining invalid characters with '-'
  name="$(printf '%s' "${name}" | sed 's/[^a-z0-9-]/-/g')"
  # Collapse consecutive dashes
  name="$(printf '%s' "${name}" | tr -s '-')"
  # Trim leading/trailing dashes
  name="${name##-}"
  name="${name%%-}"
  if [[ -z "${name}" ]]; then
    name="secret"
  fi
  # Azure maximum length is 253 characters
  if [[ ${#name} -gt 253 ]]; then
    name="${name:0:253}"
    name="${name%%-}"
  fi
  printf '%s' "${name}"
}

RESOURCE_GROUP=""
LOCATION=""
ENVIRONMENT_NAME=""
SUBSCRIPTION=""
ACR_NAME=""
WORKSPACE_NAME=""
REGISTRY_USERNAME=""
REGISTRY_PASSWORD=""
BACKEND_APP_NAME="${DEFAULT_BACKEND_IMAGE_NAME}"
FRONTEND_APP_NAME="${DEFAULT_FRONTEND_IMAGE_NAME}"
BACKEND_IMAGE=""
FRONTEND_IMAGE=""
BACKEND_IMAGE_NAME="${DEFAULT_BACKEND_IMAGE_NAME}"
FRONTEND_IMAGE_NAME="${DEFAULT_FRONTEND_IMAGE_NAME}"
BACKEND_TAG="${DEFAULT_BACKEND_TAG}"
FRONTEND_TAG="${DEFAULT_FRONTEND_TAG}"
BACKEND_PORT="${DEFAULT_BACKEND_PORT}"
FRONTEND_PORT="${DEFAULT_FRONTEND_PORT}"
SKIP_BUILD=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --resource-group)
      RESOURCE_GROUP="$2"
      shift 2
      ;;
    --location)
      LOCATION="$2"
      shift 2
      ;;
    --environment)
      ENVIRONMENT_NAME="$2"
      shift 2
      ;;
    --subscription)
      SUBSCRIPTION="$2"
      shift 2
      ;;
    --acr-name)
      ACR_NAME="$2"
      shift 2
      ;;
    --workspace-name)
      WORKSPACE_NAME="$2"
      shift 2
      ;;
    --registry-username)
      REGISTRY_USERNAME="$2"
      shift 2
      ;;
    --registry-password)
      REGISTRY_PASSWORD="$2"
      shift 2
      ;;
    --backend-app-name)
      BACKEND_APP_NAME="$2"
      shift 2
      ;;
    --frontend-app-name)
      FRONTEND_APP_NAME="$2"
      shift 2
      ;;
    --backend-image)
      BACKEND_IMAGE="$2"
      shift 2
      ;;
    --frontend-image)
      FRONTEND_IMAGE="$2"
      shift 2
      ;;
    --backend-image-name)
      BACKEND_IMAGE_NAME="$2"
      shift 2
      ;;
    --frontend-image-name)
      FRONTEND_IMAGE_NAME="$2"
      shift 2
      ;;
    --backend-tag)
      BACKEND_TAG="$2"
      shift 2
      ;;
    --frontend-tag)
      FRONTEND_TAG="$2"
      shift 2
      ;;
    --backend-port)
      BACKEND_PORT="$2"
      shift 2
      ;;
    --frontend-port)
      FRONTEND_PORT="$2"
      shift 2
      ;;
    --skip-build)
      SKIP_BUILD=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${RESOURCE_GROUP}" || -z "${LOCATION}" || -z "${ENVIRONMENT_NAME}" || -z "${ACR_NAME}" ]]; then
  echo "Error: --resource-group, --location, --environment, and --acr-name are required." >&2
  echo >&2
  usage
  exit 1
fi

if [[ ! -f "${BACKEND_ENV_FILE}" ]]; then
  echo "Error: ${BACKEND_ENV_FILE} not found. Populate .env before running the deploy script." >&2
  exit 1
fi

if [[ ! -f "${FRONTEND_ENV_FILE}" ]]; then
  echo "Error: ${FRONTEND_ENV_FILE} not found. Populate web/.env before running the deploy script." >&2
  exit 1
fi

require_command az
require_command python3

if [[ "${SKIP_BUILD}" -eq 0 ]]; then
  require_command docker
fi

if [[ -n "${SUBSCRIPTION}" ]]; then
  echo "Selecting Azure subscription ${SUBSCRIPTION}"
  az account set --subscription "${SUBSCRIPTION}"
fi

echo "Ensuring resource group ${RESOURCE_GROUP} in ${LOCATION}"
az group create --name "${RESOURCE_GROUP}" --location "${LOCATION}" >/dev/null

if [[ -z "${WORKSPACE_NAME}" ]]; then
  WORKSPACE_NAME="${ENVIRONMENT_NAME}-logs"
fi

if ! az monitor log-analytics workspace show --resource-group "${RESOURCE_GROUP}" --workspace-name "${WORKSPACE_NAME}" >/dev/null 2>&1; then
  echo "Creating Log Analytics workspace ${WORKSPACE_NAME}"
  az monitor log-analytics workspace create \
    --resource-group "${RESOURCE_GROUP}" \
    --workspace-name "${WORKSPACE_NAME}" \
    --location "${LOCATION}" >/dev/null
fi

WORKSPACE_ID="$(az monitor log-analytics workspace show \
  --resource-group "${RESOURCE_GROUP}" \
  --workspace-name "${WORKSPACE_NAME}" \
  --query customerId -o tsv)"

WORKSPACE_KEY="$(az monitor log-analytics workspace get-shared-keys \
  --resource-group "${RESOURCE_GROUP}" \
  --workspace-name "${WORKSPACE_NAME}" \
  --query primarySharedKey -o tsv)"

if ! az containerapp env show --name "${ENVIRONMENT_NAME}" --resource-group "${RESOURCE_GROUP}" >/dev/null 2>&1; then
  echo "Creating Container Apps environment ${ENVIRONMENT_NAME}"
  az containerapp env create \
    --name "${ENVIRONMENT_NAME}" \
    --resource-group "${RESOURCE_GROUP}" \
    --location "${LOCATION}" \
    --logs-workspace-id "${WORKSPACE_ID}" \
    --logs-workspace-key "${WORKSPACE_KEY}" >/dev/null
fi

ACR_LOGIN_SERVER="$(az acr show --name "${ACR_NAME}" --query loginServer -o tsv)"

if [[ -z "${ACR_LOGIN_SERVER}" ]]; then
  echo "Error: Unable to resolve login server for ACR ${ACR_NAME}" >&2
  exit 1
fi

if [[ -z "${REGISTRY_USERNAME}" || -z "${REGISTRY_PASSWORD}" ]]; then
  if az acr credential show --name "${ACR_NAME}" >/dev/null 2>&1; then
    REGISTRY_USERNAME="$(az acr credential show --name "${ACR_NAME}" --query username -o tsv)"
    REGISTRY_PASSWORD="$(az acr credential show --name "${ACR_NAME}" --query passwords[0].value -o tsv)"
  else
    cat >&2 <<EOF
Error: Registry credentials were not supplied and admin user access is disabled.
Either:
  * enable the admin user on the registry (az acr update --name ${ACR_NAME} --admin-enabled true), or
  * rerun this script with --registry-username/--registry-password, or
  * assign a managed identity with ACR pull permissions and update the script accordingly.
EOF
    exit 1
  fi
fi

if [[ -z "${BACKEND_IMAGE}" ]]; then
  BACKEND_IMAGE="${ACR_LOGIN_SERVER}/${BACKEND_IMAGE_NAME}:${BACKEND_TAG}"
fi

if [[ -z "${FRONTEND_IMAGE}" ]]; then
  FRONTEND_IMAGE="${ACR_LOGIN_SERVER}/${FRONTEND_IMAGE_NAME}:${FRONTEND_TAG}"
fi

BACKEND_ENV_VARS=()
BACKEND_SECRETS=()
declare -A BACKEND_SECRET_NAME_MAP=()
declare -A BACKEND_SECRET_NAME_USED=()
while IFS=$'\t' read -r key value; do
  if [[ -z "${key}" ]]; then
    continue
  fi
  if [[ -z "${value}" ]]; then
    continue
  fi
  if is_secret_key "${key}"; then
    secret_name="${BACKEND_SECRET_NAME_MAP[${key}]-}"
    if [[ -z "${secret_name}" ]]; then
      secret_name="$(sanitize_secret_name "${key}")"
      base_name="${secret_name}"
      suffix=1
      while [[ -n "${BACKEND_SECRET_NAME_USED[${secret_name}]-}" ]]; do
        secret_name="${base_name}-${suffix}"
        ((suffix++))
      done
      BACKEND_SECRET_NAME_MAP["${key}"]="${secret_name}"
      BACKEND_SECRET_NAME_USED["${secret_name}"]=1
    fi
    BACKEND_SECRETS+=("${secret_name}=${value}")
    BACKEND_ENV_VARS+=("${key}=secretref:${secret_name}")
  else
    BACKEND_ENV_VARS+=("${key}=${value}")
  fi
done < <(parse_env_file "${BACKEND_ENV_FILE}")

FRONTEND_ENV_VARS=()
while IFS=$'\t' read -r key value; do
  if [[ -z "${key}" ]]; then
    continue
  fi
  if [[ -z "${value}" ]]; then
    continue
  fi
  FRONTEND_ENV_VARS+=("${key}=${value}")
done < <(parse_env_file "${FRONTEND_ENV_FILE}")

if [[ "${SKIP_BUILD}" -eq 0 ]]; then
  echo "Logging into ACR ${ACR_NAME}"
  az acr login --name "${ACR_NAME}" >/dev/null

  echo "Building backend image ${BACKEND_IMAGE}"
  docker build \
    --file "${REPO_ROOT}/Dockerfile" \
    --tag "${BACKEND_IMAGE}" \
    "${REPO_ROOT}"

  echo "Pushing backend image ${BACKEND_IMAGE}"
  docker push "${BACKEND_IMAGE}"

  echo "Building frontend image ${FRONTEND_IMAGE}"
  docker build \
    --file "${REPO_ROOT}/web/Dockerfile" \
    --tag "${FRONTEND_IMAGE}" \
    "${REPO_ROOT}"

  echo "Pushing frontend image ${FRONTEND_IMAGE}"
  docker push "${FRONTEND_IMAGE}"
fi

REVISION_SUFFIX="$(date +%Y%m%d%H%M%S)"

BACKEND_CMD=(
  az containerapp create
  --name "${BACKEND_APP_NAME}"
  --resource-group "${RESOURCE_GROUP}"
  --environment "${ENVIRONMENT_NAME}"
  --image "${BACKEND_IMAGE}"
  --target-port "${BACKEND_PORT}"
  --ingress external
  --transport auto
  --cpu 1
  --memory 2Gi
  --min-replicas 1
  --max-replicas 1
  --registry-server "${ACR_LOGIN_SERVER}"
  --registry-username "${REGISTRY_USERNAME}"
  --registry-password "${REGISTRY_PASSWORD}"
  --revision-suffix "rev-${REVISION_SUFFIX}"
)

if [[ "${#BACKEND_ENV_VARS[@]}" -gt 0 ]]; then
  BACKEND_CMD+=(--env-vars "${BACKEND_ENV_VARS[@]}")
fi

if [[ "${#BACKEND_SECRETS[@]}" -gt 0 ]]; then
  BACKEND_CMD+=(--secrets "${BACKEND_SECRETS[@]}")
fi

echo "Creating backend container app ${BACKEND_APP_NAME}"
BACKEND_CREATE_LOG="$(mktemp)"
if "${BACKEND_CMD[@]}" >/dev/null 2>"${BACKEND_CREATE_LOG}"; then
  rm -f "${BACKEND_CREATE_LOG}"
else
  if grep -qi "already exists" "${BACKEND_CREATE_LOG}"; then
    rm -f "${BACKEND_CREATE_LOG}"
    echo "Backend app ${BACKEND_APP_NAME} already exists; updating instead"

    if [[ "${#BACKEND_SECRETS[@]}" -gt 0 ]]; then
      az containerapp secret set \
        --name "${BACKEND_APP_NAME}" \
        --resource-group "${RESOURCE_GROUP}" \
        --secrets "${BACKEND_SECRETS[@]}" >/dev/null
    fi

    BACKEND_UPDATE_CMD=(
      az containerapp update
      --name "${BACKEND_APP_NAME}"
      --resource-group "${RESOURCE_GROUP}"
      --image "${BACKEND_IMAGE}"
      --cpu 1
      --memory 2Gi
      --min-replicas 1
      --max-replicas 1
      --revision-suffix "rev-${REVISION_SUFFIX}"
    )

    if [[ "${#BACKEND_ENV_VARS[@]}" -gt 0 ]]; then
      BACKEND_UPDATE_CMD+=(--replace-env-vars "${BACKEND_ENV_VARS[@]}")
    fi

    "${BACKEND_UPDATE_CMD[@]}" >/dev/null
  else
    cat "${BACKEND_CREATE_LOG}" >&2
    rm -f "${BACKEND_CREATE_LOG}"
    exit 1
  fi
fi

FRONTEND_CMD=(
  az containerapp create
  --name "${FRONTEND_APP_NAME}"
  --resource-group "${RESOURCE_GROUP}"
  --environment "${ENVIRONMENT_NAME}"
  --image "${FRONTEND_IMAGE}"
  --target-port "${FRONTEND_PORT}"
  --ingress external
  --transport auto
  --cpu 0.5
  --memory 1Gi
  --min-replicas 1
  --max-replicas 1
  --registry-server "${ACR_LOGIN_SERVER}"
  --registry-username "${REGISTRY_USERNAME}"
  --registry-password "${REGISTRY_PASSWORD}"
  --revision-suffix "rev-${REVISION_SUFFIX}"
)

if [[ "${#FRONTEND_ENV_VARS[@]}" -gt 0 ]]; then
  FRONTEND_CMD+=(--env-vars "${FRONTEND_ENV_VARS[@]}")
fi

echo "Creating frontend container app ${FRONTEND_APP_NAME}"
FRONTEND_CREATE_LOG="$(mktemp)"
if "${FRONTEND_CMD[@]}" >/dev/null 2>"${FRONTEND_CREATE_LOG}"; then
  rm -f "${FRONTEND_CREATE_LOG}"
else
  if grep -qi "already exists" "${FRONTEND_CREATE_LOG}"; then
    rm -f "${FRONTEND_CREATE_LOG}"
    echo "Frontend app ${FRONTEND_APP_NAME} already exists; updating instead"

    FRONTEND_UPDATE_CMD=(
      az containerapp update
      --name "${FRONTEND_APP_NAME}"
      --resource-group "${RESOURCE_GROUP}"
      --image "${FRONTEND_IMAGE}"
      --cpu 0.5
      --memory 1Gi
      --min-replicas 1
      --max-replicas 1
      --revision-suffix "rev-${REVISION_SUFFIX}"
    )

    if [[ "${#FRONTEND_ENV_VARS[@]}" -gt 0 ]]; then
      FRONTEND_UPDATE_CMD+=(--replace-env-vars "${FRONTEND_ENV_VARS[@]}")
    fi

    "${FRONTEND_UPDATE_CMD[@]}" >/dev/null
  else
    cat "${FRONTEND_CREATE_LOG}" >&2
    rm -f "${FRONTEND_CREATE_LOG}"
    exit 1
  fi
fi

BACKEND_FQDN="$(az containerapp show --name "${BACKEND_APP_NAME}" --resource-group "${RESOURCE_GROUP}" --query properties.configuration.ingress.fqdn -o tsv)"
FRONTEND_FQDN="$(az containerapp show --name "${FRONTEND_APP_NAME}" --resource-group "${RESOURCE_GROUP}" --query properties.configuration.ingress.fqdn -o tsv)"

echo
echo "Deployment completed."
echo "Backend  FQDN: ${BACKEND_FQDN}"
echo "Frontend FQDN: ${FRONTEND_FQDN}"
echo
echo "Tip: update web/.env so VITE_API_BASE_URL points at the backend FQDN, then rerun the script to rebuild the frontend image."
