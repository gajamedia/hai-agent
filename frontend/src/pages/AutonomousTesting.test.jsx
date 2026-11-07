
import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import AutonomousTesting from "./AutonomousTesting";

vi.mock("../api/client", () => ({
  createAutonomousTestBatch: vi.fn(),
  startAutonomousTestBatch: vi.fn(),
  getAutonomousTestBatches: vi.fn().mockResolvedValue({ items: [] }),
  getAutonomousTestBatch: vi.fn(),
  getAutonomousTestBatchResults: vi.fn(),
}));

describe("AutonomousTesting", () => {
  it("renders batch configuration and history workspace", async () => {
    render(
      <MemoryRouter>
        <AutonomousTesting />
      </MemoryRouter>
    );

    expect(
      screen.getByText("Autonomous Testing Workspace")
    ).toBeTruthy();

    expect(
      screen.getByText("Buat batch pengujian")
    ).toBeTruthy();

    expect(
      screen.getByText("Riwayat batch")
    ).toBeTruthy();
  });
});
