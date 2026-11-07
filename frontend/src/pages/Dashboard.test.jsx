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
  getEnvironmentProviders: vi.fn(),
  getEnvironmentInstances: vi.fn(),
  getEnvironmentInstanceCapabilities: vi.fn(),
}));


vi.mock("../components/AssistantComposer.jsx", () => ({
  default: ({ onPrepareExecution }) => (
    <button
      type="button"
      onClick={() => onPrepareExecution({
        goalDescription: "Open the box",
        plan: {
          goal: "Open the box",
          rationale: "Use the registered action.",
          steps: [{
            index: 1,
            action: "open_box",
            purpose: "Open the box",
            preconditions: [],
            expected_observation: "The box is open.",
            success_criteria: "box=open",
            risk: null,
          }],
          assumptions: [],
          constraints: [],
          stop_conditions: [],
          confidence: 0.95,
          provider: "test-provider",
          model: "test-model",
        },
      })}
    >
      Prepare test Cognitive Pass plan
    </button>
  ),
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
  api.getEnvironmentProviders.mockResolvedValue({ items: [] });
  api.getEnvironmentInstances.mockResolvedValue({ items: [] });
  api.getEnvironmentInstanceCapabilities.mockResolvedValue({
    available_capability_ids: [],
    candidates: [],
  });
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
      name: "Run autonomous goal",
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
    expect(payload).not.toHaveProperty("actions");
    expect(payload).not.toHaveProperty("plan");
    expect(idempotencyKey).toEqual(expect.any(String));
    expect(idempotencyKey.length).toBeGreaterThan(0);
  });

  it("hands the original Cognitive Pass plan to the backend without actions", async () => {
    const user = userEvent.setup();

    renderDashboard();

    await user.click(await screen.findByRole("button", {
      name: "Prepare test Cognitive Pass plan",
    }));

    expect(
      await screen.findByText("Cognitive Pass plan siap diserahkan"),
    ).toBeInTheDocument();

    const button = screen.getByRole("button", {
      name: "Run autonomous goal",
    });
    await waitFor(() => expect(button).toBeEnabled());
    await user.click(button);

    await screen.findByText("Goal tercapai");

    const [payload] = api.runCognitivePlan.mock.calls[0];
    expect(payload.plan).toMatchObject({
      goal: "Open the box",
      provider: "test-provider",
      model: "test-model",
      steps: [{ index: 1, action: "open_box", purpose: "Open the box" }],
    });
    expect(payload).not.toHaveProperty("actions");
    expect(payload.observation).toEqual(initialState.state);
  });

  it("discards a prepared plan when the goal is edited", async () => {
    const user = userEvent.setup();

    renderDashboard();

    await user.click(await screen.findByRole("button", {
      name: "Prepare test Cognitive Pass plan",
    }));
    expect(
      await screen.findByText("Cognitive Pass plan siap diserahkan"),
    ).toBeInTheDocument();

    const goalInput = screen.getByPlaceholderText("Contoh: Unlock the door");
    await user.clear(goalInput);
    await user.type(goalInput, "Take the key");

    expect(
      screen.queryByText("Cognitive Pass plan siap diserahkan"),
    ).not.toBeInTheDocument();

    const button = screen.getByRole("button", {
      name: "Run autonomous goal",
    });
    await waitFor(() => expect(button).toBeEnabled());
    await user.click(button);
    await screen.findByText("Goal tercapai");

    const [payload] = api.runCognitivePlan.mock.calls[0];
    expect(payload).not.toHaveProperty("plan");
    expect(payload).not.toHaveProperty("actions");
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
      name: "Run autonomous goal",
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
      name: "Run autonomous goal",
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

    // An unknown outcome must block every UI action that can mutate simulation state.
    expect(
      screen.getByRole("button", { name: "Run autonomous goal" }),
    ).toBeDisabled();
    expect(
      screen.queryByRole("button", { name: "Execute actions" }),
    ).not.toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Reset simulation" }),
    ).toBeDisabled();

    // Reading state remains available; it does not clear the safety lock.
    expect(
      screen.getByRole("button", { name: "Refresh state" }),
    ).toBeEnabled();
  });

  it("lets manual mode send the explicit selected action IDs", async () => {
    const user = userEvent.setup();

    renderDashboard();

    const manualMode = await screen.findByRole("button", {
      name: "Manual mode",
    });
    await user.click(manualMode);

    const button = screen.getByRole("button", { name: "Plan & Run Goal" });
    await waitFor(() => expect(button).toBeEnabled());
    await user.click(button);

    await screen.findByText("Goal tercapai");

    const [payload] = api.runCognitivePlan.mock.calls[0];
    expect(payload.actions).toEqual(["open_box"]);
    expect(payload).not.toHaveProperty("plan");
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
      name: "Run autonomous goal",
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
