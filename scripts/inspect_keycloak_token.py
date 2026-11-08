import base64
import getpass
import json


def decode_payload(token: str) -> dict:
    parts = token.strip().split(".")

    if len(parts) != 3:
        raise ValueError("Format JWT tidak valid.")

    payload = parts[1]
    payload += "=" * (-len(payload) % 4)

    decoded = base64.urlsafe_b64decode(payload)
    return json.loads(decoded)


def main():
    token = getpass.getpass("Paste access token (hidden): ")

    try:
        claims = decode_payload(token)

        print("\n=== KEYCLOAK ACCESS TOKEN CLAIMS ===")
        print("Issuer:", claims.get("iss"))
        print("Audience:", claims.get("aud"))
        print("Token type:", claims.get("typ"))
        print("Preferred username:", claims.get("preferred_username"))

        print("\n=== REALM ROLES ===")
        realm_access = claims.get("realm_access") or {}
        print(json.dumps(
            realm_access.get("roles", []),
            indent=2
        ))

        print("\n=== API CLIENT ROLES ===")
        resource_access = claims.get("resource_access") or {}
        api_access = resource_access.get("hai-agent-api") or {}

        print(json.dumps(
            api_access.get("roles", []),
            indent=2
        ))

        print("\n=== EXPLICIT PERMISSIONS ===")
        print(json.dumps(
            claims.get("permissions", []),
            indent=2
        ))

    except Exception as exc:
        print("Token inspection failed:", str(exc))


if __name__ == "__main__":
    main()
