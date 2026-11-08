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
          <Route index element={<Navigate to="reports" replace />} />
          <Route path="reports" element={<h1>LAPORAN PAGE</h1>} />
          <Route path="changes" element={<h1>UBAH PAGE</h1>} />
          <Route path="create" element={<h1>BUAT PAGE</h1>} />
        </Route>
        <Route path="/data-changes" element={<Navigate to="/database-agent/changes" replace />} />
      </Routes>
    </MemoryRouter>,
  );
}

describe("Database Agent menu", () => {
  it("offers the three tabs", () => {
    renderAt("/database-agent/reports");
    const nav = screen.getByRole("navigation", { name: "Database Agent" });
    expect(nav.querySelectorAll("a")).toHaveLength(DATABASE_AGENT_TABS.length);
    for (const label of ["Laporan", "Ubah data", "Buat database"]) {
      expect(screen.getByRole("link", { name: label })).toBeTruthy();
    }
  });

  it("opens Laporan by default", async () => {
    renderAt("/database-agent");
    expect(await screen.findByText("LAPORAN PAGE")).toBeTruthy();
  });

  it("switches tabs without leaving the menu", async () => {
    const user = userEvent.setup();
    renderAt("/database-agent/reports");
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
