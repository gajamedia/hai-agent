import Keycloak from "keycloak-js";

const keycloakConfig = {
  url: (import.meta.env.VITE_KEYCLOAK_URL || "http://localhost:8080").replace(/\/+$/, ""),
  realm: import.meta.env.VITE_KEYCLOAK_REALM || "hai-agent",
  clientId: import.meta.env.VITE_KEYCLOAK_CLIENT_ID || "hai-agent-web",
};

const keycloak = new Keycloak(keycloakConfig);
let initializationPromise = null;

/**
 * Where Keycloak should send the browser back after signing in: the page the user was on.
 * A fixed "/" made every full page load (F5, a plain link, a bookmark) land on the home
 * dashboard instead of the page that was asked for. Only same-origin location parts are
 * used, and Keycloak still validates the result against the client's allowed redirect URIs.
 */
export function currentPageRedirectUri() {
  const { origin, pathname, search } = window.location;
  return `${origin}${pathname || "/"}${search}`;
}

export function getKeycloak() {
  return keycloak;
}

export function initializeKeycloak() {
  if (!initializationPromise) {
    initializationPromise = keycloak.init({
      onLoad: "login-required",
      flow: "standard",
      pkceMethod: "S256",
      checkLoginIframe: false,
      redirectUri: currentPageRedirectUri(),
    }).catch((error) => {
      // Permit an explicit retry after an initialization failure.
      initializationPromise = null;
      throw error;
    });
  }

  return initializationPromise;
}

export async function getFreshAccessToken() {
  if (!keycloak.authenticated) {
    throw new Error("Belum login ke Keycloak. Silakan login kembali.");
  }

  // Refresh only when the token is expired or will expire within 30 seconds.
  await keycloak.updateToken(30);

  if (!keycloak.token) {
    throw new Error("Keycloak tidak mengembalikan access token.");
  }

  return keycloak.token;
}

export function getAuthenticatedUser() {
  const claims = keycloak.tokenParsed || {};
  return {
    username:
      claims.preferred_username ||
      claims.email ||
      claims.name ||
      "Authenticated user",
    email: claims.email || "",
  };
}

export async function logoutFromKeycloak() {
  await keycloak.logout({
    redirectUri: `${window.location.origin}/`,
  });
}
