import { afterEach, describe, expect, it } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Navigate, Route, Routes } from "react-router-dom";
import DatabaseAgent, { DATABASE_AGENT_TABS } from "./DatabaseAgent";

afterEach(cleanup);

function renderAt(path) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <Routes>
        <Route path="/database-agent" element={<DatabaseAgent />}>
          <Route index element={<Navigate to="view" replace />} />
          <Route path="connections" element={<h1>KONEKSI PAGE</h1>} />
          <Route path="view" element={<h1>LIHAT PAGE</h1>} />
          <Route path="entry" element={<h1>CATAT PAGE</h1>} />
          <Route path="changes" element={<h1>UBAH PAGE</h1>} />
          <Route path="create" element={<h1>BUAT PAGE</h1>} />
        </Route>
        <Route path="/data-changes" element={<Navigate to="/database-agent/changes" replace />} />
      </Routes>
    </MemoryRouter>,
  );
}

describe("Database Agent menu", () => {
  it("offers the tabs in working order: assistant, create, connect, record, change, view", () => {
    renderAt("/database-agent/view");
    const nav = screen.getByRole("navigation", { name: "Database Agent" });
    expect(nav.querySelectorAll("a")).toHaveLength(DATABASE_AGENT_TABS.length);
    const labels = [...nav.querySelectorAll("a")].map((a) => a.textContent);
    expect(labels).toEqual(["Asisten", "Buat database", "Koneksi", "Catat data", "Ubah data", "Lihat data"]);
  });

  it("opens Lihat data by default", async () => {
    renderAt("/database-agent");
    expect(await screen.findByText("LIHAT PAGE")).toBeTruthy();
  });

  it("switches tabs without leaving the menu", async () => {
    const user = userEvent.setup();
    renderAt("/database-agent/view");
    await user.click(screen.getByRole("link", { name: "Ubah data" }));
    expect(await screen.findByText("UBAH PAGE")).toBeTruthy();
    await user.click(screen.getByRole("link", { name: "Buat database" }));
    expect(await screen.findByText("BUAT PAGE")).toBeTruthy();
    expect(screen.getByRole("navigation", { name: "Database Agent" })).toBeTruthy();
  });

  it("keeps old bookmarks working", async () => {
    renderAt("/data-changes");
    expect(await screen.findByText("UBAH PAGE")).toBeTruthy();
  });
});
