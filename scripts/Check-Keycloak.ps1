[CmdletBinding()]
param(
    [string]$KeycloakBaseUrl = "http://localhost:8080",
    [string]$Realm = "hai-agent",
    [string]$ApiBaseUrl = "http://127.0.0.1:8000",
    [string]$AccessToken = $env:HAI_KEYCLOAK_TEST_ACCESS_TOKEN
)

$ErrorActionPreference = "Stop"
$base = $KeycloakBaseUrl.TrimEnd("/")
$issuer = "$base/realms/$Realm"
$discoveryUrl = "$issuer/.well-known/openid-configuration"

Write-Host "`n=== KEYCLOAK OIDC DISCOVERY ===" -ForegroundColor Cyan
$discovery = Invoke-RestMethod -Method Get -Uri $discoveryUrl
if ($discovery.issuer.TrimEnd("/") -ne $issuer.TrimEnd("/")) {
    throw "Issuer mismatch. Expected '$issuer', received '$($discovery.issuer)'."
}
Write-Host "Issuer: $($discovery.issuer)"
Write-Host "JWKS:   $($discovery.jwks_uri)"

Write-Host "`n=== JWKS SIGNING KEYS ===" -ForegroundColor Cyan
$jwks = Invoke-RestMethod -Method Get -Uri $discovery.jwks_uri
$signingKeys = @($jwks.keys | Where-Object {
    $_.use -eq "sig" -and $_.kty -in @("RSA", "EC") -and $_.kid
})
if ($signingKeys.Count -lt 1) {
    throw "No usable asymmetric signing keys found in JWKS."
}
$signingKeys | ForEach-Object {
    Write-Host "kid=$($_.kid) kty=$($_.kty) alg=$($_.alg)"
}

Write-Host "`n=== API TOKEN VALIDATION (OPTIONAL) ===" -ForegroundColor Cyan
if ([string]::IsNullOrWhiteSpace($AccessToken)) {
    Write-Host "Skipped. Set HAI_KEYCLOAK_TEST_ACCESS_TOKEN to a real access token to test API middleware."
    exit 0
}

$headers = @{ Authorization = "Bearer $AccessToken" }
$uri = "$($ApiBaseUrl.TrimEnd('/'))/api/v1/capability-proposals?limit=1"
try {
    $response = Invoke-WebRequest -Method Get -Uri $uri -Headers $headers
    Write-Host "API returned HTTP $([int]$response.StatusCode). Bearer authentication passed."
}
catch {
    $status = 0
    if ($_.Exception.Response) {
        $status = [int]$_.Exception.Response.StatusCode
    }
    $body = ""
    try {
        $stream = $_.Exception.Response.GetResponseStream()
        if ($stream) {
            $reader = [System.IO.StreamReader]::new($stream)
            $body = $reader.ReadToEnd()
            $reader.Dispose()
        }
    } catch {}

    if ($status -in @(200, 403)) {
        Write-Host "API returned HTTP $status. Bearer authentication passed; authorization may deny this principal."
        exit 0
    }
    if ($status -eq 503 -and $body -notmatch "authentication_unavailable") {
        Write-Host "API returned HTTP 503 after authentication. Check whether PBAC policy/approval services are configured."
        exit 0
    }
    throw "API token validation did not pass. HTTP $status. $body"
}
