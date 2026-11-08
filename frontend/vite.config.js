import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],

  server: {
    host: "127.0.0.1",
    port: 5173,
  },

  test: {
    environment: "jsdom",

    // Vitest hanya menjalankan unit test, bukan Playwright E2E.
    include: [
      "src/**/*.test.{js,jsx,ts,tsx}",
      "src/**/*.spec.{js,jsx,ts,tsx}",
    ],

    exclude: [
      "node_modules/**",
      "dist/**",
      "e2e/**",
      "playwright-report/**",
      "test-results/**",
    ],

    clearMocks: true,
    restoreMocks: true,
  },
});
