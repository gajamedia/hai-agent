import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import { setAccessTokenProvider } from "../api/client.js";
import {
  getAuthenticatedUser,
  getFreshAccessToken,
  getKeycloak,
  initializeKeycloak,
  logoutFromKeycloak,
} from "./keycloak.js";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [status, setStatus] = useState("initializing");
  const [user, setUser] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;

    initializeKeycloak()
      .then((authenticated) => {
        if (!authenticated || !getKeycloak().authenticated) {
          throw new Error("Login Keycloak belum berhasil.");
        }

        setAccessTokenProvider(getFreshAccessToken);

        if (active) {
          setUser(getAuthenticatedUser());
          setStatus("authenticated");
          setError("");
        }
      })
      .catch((cause) => {
        if (active) {
          setAccessTokenProvider(null);
          setError(
            cause?.message ||
              "Tidak dapat menginisialisasi autentikasi Keycloak.",
          );
          setStatus("error");
        }
      });

    return () => {
      active = false;
      // Do not clear the provider on React StrictMode's development
      // effect cleanup; the module-level Keycloak initialization is shared.
    };
  }, []);

  const login = useCallback(() => {
    setError("");
    return getKeycloak().login({
      redirectUri: `${window.location.origin}/`,
    });
  }, []);

  const logout = useCallback(async () => {
    setAccessTokenProvider(null);
    await logoutFromKeycloak();
  }, []);

  const value = useMemo(
    () => ({ status, user, error, login, logout }),
    [status, user, error, login, logout],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const value = useContext(AuthContext);
  if (!value) {
    throw new Error("useAuth must be used inside AuthProvider.");
  }
  return value;
}
