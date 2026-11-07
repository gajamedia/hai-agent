import "@testing-library/jest-dom/vitest";
import React from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

const api = vi.hoisted(() => ({
  getEnvironmentProviders: vi.fn(),
  getEnvironmentInstances: vi.fn(),
  getEnvironmentInstanceCapabilities: vi.fn(),
}));

vi.mock("../api/client.js", () => api);

import EnvironmentWorkspace from "./EnvironmentWorkspace.jsx";

const providers = {
  items: [{
    provider_id: "sim-provider",
    descriptor: {
      environment_type: "room",
      domain: "simulation",
      mode: "simulation",
    },
    active_instance_count: 1,
  }],
};

const instances = {
  items: [{
    instance_id: "room-001",
    provider_id: "sim-provider",
    status: "active",
    descriptor: { environment_type: "room" },
  }],
};

const capabilityReport = {
  environment_instance_id: "room-001",
  environment_type: "room",
  registry_version: 7,
  available_capability_ids: ["cap.open"],
  candidates: [
    {
      capability_id: "cap.open",
      action_name: "open_box",
      description: "Open a box",
      risk_level: "low",
      permissions: ["room:write"],
      state: "available",
    },
    {
      capability_id: "cap.disabled",
      action_name: "take_key",
      description: "Take a key",
      risk_level: "medium",
      permissions: ["room:write"],
      state: "disabled",
    },
  ],
};

beforeEach(() => {
  vi.clearAllMocks();
  api.getEnvironmentProviders.mockResolvedValue(providers);
  api.getEnvironmentInstances.mockResolvedValue(instances);
  api.getEnvironmentInstanceCapabilities.mockResolvedValue(capabilityReport);
});

afterEach(cleanup);

describe("EnvironmentWorkspace", () => {
  it("loads trusted providers, active instances, and capability metadata", async () => {
    render(<EnvironmentWorkspace />);

    expect(await screen.findByText("open_box")).toBeInTheDocument();
    expect(screen.getByText("room-001")).toBeInTheDocument();
    expect(api.getEnvironmentProviders).toHaveBeenCalledTimes(1);
    expect(api.getEnvironmentInstances).toHaveBeenCalledTimes(1);
    expect(api.getEnvironmentInstanceCapabilities).toHaveBeenCalledWith(
      "room-001",
      expect.any(AbortSignal),
    );
  });

  it("allows selection only for available and unambiguous capabilities", async () => {
    const user = userEvent.setup();
    render(<EnvironmentWorkspace />);

    const available = await screen.findByRole("checkbox", { name: /open_box/ });
    const disabled = screen.getByRole("checkbox", { name: /take_key/ });

    expect(available).toBeEnabled();
    expect(disabled).toBeDisabled();

    await user.click(available);
    expect(available).toBeChecked();
    expect(screen.getByText("1 dipilih")).toBeInTheDocument();
  });

  it("shows an empty state when no provider is registered", async () => {
    api.getEnvironmentProviders.mockResolvedValue({ items: [] });
    api.getEnvironmentInstances.mockResolvedValue({ items: [] });

    render(<EnvironmentWorkspace />);

    expect(await screen.findByText("Belum ada environment provider")).toBeInTheDocument();
    expect(api.getEnvironmentInstanceCapabilities).not.toHaveBeenCalled();
  });

  it("shows capability discovery errors without exposing execution controls", async () => {
    api.getEnvironmentInstanceCapabilities.mockRejectedValue(new Error("Discovery unavailable"));

    render(<EnvironmentWorkspace />);

    expect(await screen.findByText(/Discovery unavailable/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /execute/i })).not.toBeInTheDocument();
  });

  it("prevents selection when multiple available capabilities share one action name", async () => {
    api.getEnvironmentInstanceCapabilities.mockResolvedValue({
      ...capabilityReport,
      available_capability_ids: ["cap.open", "cap.open.alt"],
      candidates: [
        capabilityReport.candidates[0],
        {
          ...capabilityReport.candidates[0],
          capability_id: "cap.open.alt",
        },
      ],
    });

    render(<EnvironmentWorkspace />);

    const checkboxes = await screen.findAllByRole("checkbox", { name: /open_box/ });
    expect(checkboxes).toHaveLength(2);
    expect(checkboxes.every((checkbox) => checkbox.disabled)).toBe(true);
    expect(screen.getAllByText(/Binding ambigu/)).toHaveLength(2);
  });
});
