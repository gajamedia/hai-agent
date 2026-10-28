
import "@testing-library/jest-dom/vitest";
import React from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";

const api = vi.hoisted(() => ({
  checkHealth: vi.fn(),
  getCapabilityTelemetrySources: vi.fn(),
  getExecutionState: vi.fn(),
  getStrategies: vi.fn(),
  runCognitive: vi.fn(),
  runCognitivePlan: vi.fn(),
  runExecution: vi.fn(),
  resetExecution: vi.fn(),
  getCapabilityTelemetry: vi.fn(),
}));

vi.mock("../api/client.js", () => ({
  API_BASE_URL: "http://127.0.0.1:8000",
  ...api,
}));

import Dashboard from "./Dashboard.jsx";

const initialState = {
  status: "ok",
  state: {
    box: "closed",
    key: "inside_box",
    door: "locked",
  },
  allowed_actions: [
    "open_box",
    "take_key",
    "find_spare_key",
    "unlock_door",
  ],
};

function successfulPlanRun({
  goalSatisfied = true,
  executionSuccess = true,
} = {}) {
  return {
    mode: "cognitive_plan_execution",
    success: executionSuccess && goalSatisfied,
    goal: {
      description: "Open the box",
      satisfied: goalSatisfied,
    },
    execution: {
      success: executionSuccess,
      steps: [
        {
          action: "open_box",
          success: executionSuccess,
        },
      ],
    },
    plan: {
      steps: [
        {
          action: "open_box",
          description: "Open the box",
        },
      ],
    },
    state_before: { box: "closed" },
    state_after: {
      box: goalSatisfied ? "open" : "closed",
    },
  };
}

function renderDashboard() {
  return render(
    <MemoryRouter initialEntries={["/"]}>
      <Dashboard />
    </MemoryRouter>,
  );
}

beforeEach(() => {
  vi.clearAllMocks();

  api.checkHealth.mockResolvedValue({ status: "ok" });
  api.getCapabilityTelemetrySources.mockResolvedValue({});
  api.getCapabilityTelemetry.mockResolvedValue({});
  api.getStrategies.mockResolvedValue([]);
  api.getExecutionState.mockResolvedValue(initialState);
  api.runCognitive.mockResolvedValue({});
  api.runExecution.mockResolvedValue({});
  api.resetExecution.mockResolvedValue(initialState);
  api.runCognitivePlan.mockResolvedValue(successfulPlanRun());
});

afterEach(() => {
  cleanup();
});

describe("Dashboard goal execution", () => {
  it("shows goal verification as passed when the goal is satisfied", async () => {
    const user = userEvent.setup();

    renderDashboard();

    const button = await screen.findByRole("button", {
      name: "Plan & Run Goal",
    });

    await waitFor(() => {
      expect(button).toBeEnabled();
    });

    await user.click(button);

    expect(await screen.findByText("Goal tercapai")).toBeInTheDocument();

    expect(screen.getByText("GOAL VERIFICATION")).toBeInTheDocument();
    expect(screen.getByText("Lulus")).toBeInTheDocument();

    expect(api.runCognitivePlan).toHaveBeenCalledTimes(1);

    const [payload, signal, idempotencyKey] =
      api.runCognitivePlan.mock.calls[0];

    expect(payload.goal_description).toBe("Open the box");
    expect(payload.target_state).toEqual({ box: "open" });
    expect(payload.actions).toContain("open_box");
    expect(idempotencyKey).toEqual(expect.any(String));
    expect(idempotencyKey.length).toBeGreaterThan(0);
  });

  it("does not claim the goal is achieved when execution succeeds but verification fails", async () => {
    const user = userEvent.setup();

    api.runCognitivePlan.mockResolvedValue(
      successfulPlanRun({
        goalSatisfied: false,
        executionSuccess: true,
      }),
    );

    renderDashboard();

    const button = await screen.findByRole("button", {
      name: "Plan & Run Goal",
    });

    await waitFor(() => expect(button).toBeEnabled());

    await user.click(button);

    expect(
      await screen.findByText("Aksi selesai, goal belum tercapai"),
    ).toBeInTheDocument();

    expect(screen.getByText("Belum lulus")).toBeInTheDocument();
    expect(screen.queryByText("Goal tercapai")).not.toBeInTheDocument();
  });

  it("shows a reconciliation warning for an unknown execution outcome", async () => {
    const user = userEvent.setup();

    const error = new Error("Execution outcome is unknown.");

    error.status = 409;
    error.body = {
      detail: {
        error: "idempotency_outcome_unknown",
        message: "Execution outcome is unknown.",
        reconciliation_required: true,
        retry_safe: false,
      },
    };

    api.runCognitivePlan.mockRejectedValue(error);

    renderDashboard();

    const button = await screen.findByRole("button", {
      name: "Plan & Run Goal",
    });

    await waitFor(() => expect(button).toBeEnabled());

    await user.click(button);

    expect(
      await screen.findByText(
        /Status eksekusi belum dapat dipastikan\./,
      ),
    ).toBeInTheDocument();

    expect(
      screen.getByText(/Jangan jalankan ulang dengan key baru\./),
    ).toBeInTheDocument();

    expect(
      screen.getByText(/rekonsiliasi diperlukan/i),
    ).toBeInTheDocument();
  });

  it("disables the plan-run button while a request is pending", async () => {
    const user = userEvent.setup();

    let resolveRequest;

    api.runCognitivePlan.mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveRequest = resolve;
        }),
    );

    renderDashboard();

    const button = await screen.findByRole("button", {
      name: "Plan & Run Goal",
    });

    await waitFor(() => expect(button).toBeEnabled());

    await user.click(button);

    await waitFor(() => {
      expect(
        screen.getByRole("button", {
          name: "Merencanakan & menjalankan...",
        }),
      ).toBeDisabled();
    });

    expect(api.runCognitivePlan).toHaveBeenCalledTimes(1);

    resolveRequest(successfulPlanRun());

    await screen.findByText("Goal tercapai");
  });
});
