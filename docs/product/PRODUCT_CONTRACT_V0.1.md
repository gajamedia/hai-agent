# hAI Workbench --- Product Contract v0.1

-   **Status:** Draft for review
-   **Project:** `new-hAI-agent` (`hai-agent`)
-   **Contract version:** 0.1
-   **Date:** 18 October 2026
-   **Scope:** Product behavior, runtime boundaries, safety
    expectations, acceptance evidence, and known gaps
-   **Important:** This document records the current verified scope and
    proposed product commitments. It does not claim production
    readiness.

## 1. Product vision

hAI Workbench is a general-purpose AI Agent platform where a user
describes a goal in natural language and the agent helps interpret the
situation, form a plan, execute permitted actions under policy, verify
outcomes against evidence, and explain what succeeded, failed, or
remains unresolved.

The product should expose the relationship between: 1. the user's goal;
2. the agent's interpretation and plan; 3. actions actually attempted;
4. policy decisions and required approvals; 5. observed state changes
and verification evidence; and 6. the final outcome and any recovery or
follow-up required.

A generated plan is not an execution record. An execution record is not,
by itself, proof that the user's goal was achieved.

## 2. Product principles

1.  **Goal-oriented:** Workflows start from a user goal or explicit
    instruction.
2.  **Supervised autonomy:** The agent may act only within the active
    policy. Actions requiring approval must not proceed without valid
    approval.
3.  **Explicit runtime boundary:** Cognitive reasoning/planning and
    action execution remain separate responsibilities.
4.  **Evidence-based completion:** A goal is complete only when the
    applicable verifier confirms it from observed state or other trusted
    evidence.
5.  **Fail closed:** Missing, invalid, or failed authorization checks
    must not grant permission to execute.
6.  **Traceability:** Important decisions, approvals, actions, errors,
    and verification outcomes must be traceable.
7.  **Idempotency where needed:** Duplicate delivery or retry must not
    create duplicate harmful side effects. Guarantees must state their
    scope and storage/transaction assumptions.
8.  **Honest scope:** Simulation evidence must never be presented as
    proof of production connector or real-world performance.
9.  **No hidden execution:** A cognitive-only request must not execute
    its generated plan.
10. **Incremental validation:** New capabilities are accepted only with
    explicit tests, evidence, and documented limitations.

## 3. Intended product surfaces

### 3.1 Web dashboard

The dashboard should provide: - natural-language task or observation
input; - interpretation and generated-plan display; - an explicit
distinction between proposed and executed actions; - execution controls
appropriate to the environment and policy; - task status and step-level
outcomes; - goal-verification result and supporting evidence; - approval
prompts and decisions when required; - runtime trace, audit, and
capability telemetry where available; - clear failure, recovery, and
unresolved-state explanations.

### 3.2 API and SDK

The HTTP API is mounted under `/api/v1` in the current application. A
stable public API/SDK contract is a product goal, but full compatibility
guarantees, authentication/authorization requirements, and SDK behavior
remain to be specified and verified.

Public API documentation should define request and response schemas,
error shapes, authorization requirements, idempotency behavior,
versioning/compatibility policy, and which operations are cognitive-only
versus state-changing.

## 4. Runtime and environment boundaries

### 4.1 Cognitive Runtime

The Cognitive Runtime interprets input and produces cognitive results,
such as interpretation, reasoning summaries, and plans. The current
`POST /api/v1/cognitive/run` endpoint is documented and implemented as a
cognitive pass that does not execute the generated plan.

A successful cognitive response means that a cognitive result was
returned. It does **not** mean that actions ran or that the goal was
achieved.

### 4.2 Execution Runtime

The Execution Runtime is responsible for explicitly requested action
execution, enforcement of execution policy, reporting action outcomes,
and observing resulting state.

The currently audited execution environment is an allowlisted
`RoomEnvironment` simulation. Its registered actions include `open_box`,
`take_key`, `find_spare_key`, and `unlock_door`. These are simulation
capabilities, not external integrations.

### 4.3 Simulation versus production

Room simulation is the current MVP execution environment. It must be
labelled as simulation in documentation, UI, benchmark reports, and
acceptance evidence.

No claim of controlling real devices, external services, production
accounts, or arbitrary environments may be made until a connector has
its own contract, security review, idempotency strategy, integration
tests, and acceptance evidence.

## 5. Task lifecycle and outcome semantics

The target product lifecycle is:

1.  **Submitted:** A user supplies a goal and relevant context.
2.  **Interpreted:** The Cognitive Runtime returns its interpretation.
3.  **Planned:** The agent produces a plan with steps and success
    criteria where supported.
4.  **Policy checked:** Each action is checked against capability, risk,
    and approval policy before execution.
5.  **Awaiting approval:** The task pauses if policy requires a valid
    approval.
6.  **Executing:** Approved and permitted actions are attempted through
    the Execution Runtime.
7.  **Verifying:** The system checks observed state or trusted evidence
    against the target condition.
8.  **Completed:** The verifier confirms the goal condition.
9.  **Failed / partially completed / blocked:** The system reports what
    did not complete and why, without claiming success.
10. **Recovery or follow-up:** Where supported and safe, the system may
    attempt recovery or ask the user for direction.

This is the intended end-to-end product contract. The current dashboard
has a cognitive workspace and a separate execution workspace; full task
lifecycle integration is not yet verified.

## 6. Approval and security requirements

-   Risk classification and authorization must be enforced by trusted
    backend policy, not by user-controlled request metadata alone.
-   A high-risk action must be blocked unless the required trusted
    approval is valid.
-   Actions explicitly marked as requiring approval remain blocked
    without approval, even if otherwise classified as low risk.
-   Unknown or unregistered actions must fail closed.
-   An authorization checker failure must not authorize execution.
-   Audit persistence failure must follow the configured fail-closed
    policy; it must not silently be treated as successful auditing.
-   CORS configuration is not authentication or authorization.
-   API authentication, identity, role/tenant boundaries, and production
    authorization are not established by the current audit and must be
    specified before multi-user or production exposure.

These requirements are consistent with the audited simulation security
boundary and its tests. They do not certify an external connector or
production deployment.

## 7. Goal verification and truthful reporting

-   Every executable task should have a verifiable completion condition
    appropriate to the environment.
-   For the current Room simulation, target-state verification uses
    non-empty target state and exact matching of the specified key/value
    conditions.
-   Report separately: plan generated, action attempted, action
    succeeded, state changed, and goal verified.
-   If verification is unavailable, inconclusive, or fails, the system
    must not report the goal as completed.
-   Verification evidence should identify the relevant observed state or
    trusted evidence source.
-   Verifier coverage and known limitations must be documented for each
    environment/connector.

## 8. Idempotency and side-effect integrity

The repository contains a SQLite idempotent side-effect store whose
documented guarantee depends on the side effect and ledger write
participating in the same SQLite transaction. It does not provide
exactly-once behavior for arbitrary external HTTP, API, or filesystem
side effects without downstream support or an appropriate
transactional/outbox/inbox design.

The current audit did not find the store integrated into the audited
runtime/API execution path. Existing benchmark duplicate-effect
callbacks count effects in isolated simulation or in-memory fixtures.
Therefore: - fixture-level zero-duplicate results are valid only within
their stated test scope; - they do not prove durable runtime/API
idempotency; - integration of idempotency into the relevant execution
path is an identified gap; - any external connector must document its
idempotency key, deduplication scope, retry behavior, and failure modes.

## 9. Traceability, audit, and learning

The system should preserve traceable evidence for cognitive decisions,
execution steps, policy decisions, approval outcomes, errors, and goal
verification.

Trace-event deduplication is distinct from action-side-effect
idempotency. A test proving that duplicate trace events produce one
trace record does not prove that a repeated action cannot happen twice.

Learning from experience must not bypass the security policy,
manufacture execution evidence, or convert an unverified outcome into
success. Learning/generalization claims must remain limited to the
evaluated tasks and environments.

## 10. Current API and dashboard inventory

The following is based on the repository files and test evidence
inspected during P0.1 reconciliation.

  ----------------------------------------------------------------------------------------------
  Capability                                     Current evidence        Contract interpretation
  ---------------------------------------------- ----------------------- -----------------------
  `GET /health`                                  Registered in           Health endpoint; not a
                                                 `api/main.py`           readiness/security
                                                                         certification

  `POST /api/v1/cognitive/run`                   Route and               Cognitive-only; must
                                                 request/response        not execute the plan
                                                 schemas audited         

  `GET /api/v1/execution/state`                  Execution route and     Reads current
                                                 frontend client audited simulation state

  `POST /api/v1/execution/run`                   Route and request       Explicit simulation
                                                 schema audited          execution of requested
                                                                         actions/plan

  `POST /api/v1/execution/plan-run`              Backend route audited   Generates a plan and
                                                                         executes it in
                                                                         simulation with
                                                                         verification; frontend
                                                                         client function was not
                                                                         found in the inspected
                                                                         client

  `POST /api/v1/execution/reset`                 Route and frontend      Resets simulation state
                                                 client audited          

  `GET /api/v1/telemetry/capabilities`           Route and client        Reads capability
                                                 audited                 telemetry

  `GET /api/v1/telemetry/capabilities/sources`   Route and client        Reports telemetry
                                                 audited                 sources separately

  `GET /api/v1/strategies`                       Route and client        Exposes strategy
                                                 audited                 information and
                                                                         simulation snapshot

  Dashboard cognitive workspace                  Component inspected     Displays cognitive
                                                                         response and generated
                                                                         plan, explicitly says
                                                                         plan has not executed

  Dashboard execution workspace                  Component partially     Allows explicit
                                                 inspected               simulation action
                                                                         selection and
                                                                         goal-state
                                                                         configuration

  Public SDK/compatibility guarantees            Not established         Gap; specify and verify
                                                                         before claiming a
                                                                         stable public contract
  ----------------------------------------------------------------------------------------------

## 11. Acceptance criteria

### 11.1 P0 benchmark baseline

The agreed baseline is 30 tasks × 3 runs = 90 runs, using the declared
benchmark fixtures.

  -----------------------------------------------------------------------
  Metric                                                           Target
  ------------------------------ ----------------------------------------
  Correct task completion                                    At least 85%

  Deterministic goal                                                 100%
  verification                   

  Unauthorized high-risk actions                                        0

  Duplicate harmful side effects                                        0

  Audit and approval evidence                 100% for applicable records
  completeness                   

  Trace integrity failures                                              0

  Recovery on recoverable        At least 80%; N/A only if no recoverable
  failures                                           failures are present
  -----------------------------------------------------------------------

Every report must disclose the environment, task set, run count, failure
injection (if any), measurement method, and limitations. Passing
fixture-based acceptance is not proof of production readiness or
broad-domain generalization.

### 11.2 Existing evidence recorded during P0.1

-   A 90-run baseline report was previously audited: 90/90 correct
    completion, deterministic verification, audit evidence, approval
    evidence, and trace integrity; unauthorized actions and duplicate
    harmful side effects were zero in that fixture scope.
-   No recoverable failures were present in that baseline, so recovery
    was N/A.
-   A controlled transient-failure injection report recorded 90/90
    recoveries (100%) in the controlled Room fixture.
-   The full test suite was previously reported as 3,234 passed with one
    Starlette deprecation warning. This is a historical result, not a
    claim that tests were rerun for this document.
-   Focused Step 31/32/33, execution-security, P0 acceptance, and LLM
    contract tests were previously reported passing.
-   Step 30 and Step 30.5 adaptation results were measured on small
    deterministic Room fixtures only; they do not establish
    generalization across domains.

## 12. Confirmed gaps and required follow-up

  ------------------------------------------------------------------------------
  ID                      Gap                            Required follow-up
  ----------------------- ------------------------------ -----------------------
  G-01                    Idempotency store not found    Define idempotency
                          integrated into runtime/API    scope and integrate
                          path                           with execution path;
                                                         test duplicate
                                                         delivery, retry,
                                                         restart, conflict, and
                                                         rollback

  G-02                    Dashboard has separate         Define task lifecycle
                          cognitive and execution        UI and explicit
                          workspaces; end-to-end         transition/approval
                          lifecycle integration not      semantics before
                          verified                       implementation

  G-03                    Frontend client does not       Decide whether/how UI
                          expose `/execution/plan-run`   should call the
                          in the inspected file          endpoint; ensure
                                                         approval, verification,
                                                         and truthful status
                                                         presentation

  G-04                    API                            Specify identity,
                          authentication/authorization   permissions, trust
                          for multi-user or production   boundaries, secrets
                          use not established by this    handling, and
                          audit                          deployment controls

  G-05                    Public API/SDK compatibility   Define versioning,
                          policy not established         schema stability,
                                                         errors, deprecation,
                                                         and idempotency
                                                         contract

  G-06                    External connectors are not    Require
                          part of the current Room       connector-specific
                          simulation contract            risk, approval,
                                                         idempotency, audit,
                                                         verification, and
                                                         integration acceptance

  G-07                    Current benchmark evidence is  Expand benchmark
                          fixture-scoped                 coverage only with
                                                         declared environments,
                                                         representative tasks,
                                                         and reproducible
                                                         evidence
  ------------------------------------------------------------------------------

## 13. Non-goals for this contract version

-   Claiming production readiness based only on Room simulation tests.
-   Claiming universal or broad-domain agent generalization from
    deterministic fixture benchmarks.
-   Treating generated plans as completed actions.
-   Treating trace deduplication as side-effect idempotency.
-   Treating CORS as API authentication.
-   Enabling arbitrary action execution or allowing the agent to bypass
    trusted approval policy.
-   Integrating real external services before their security and
    execution contracts are defined.

## 14. Change control and readiness

This document is a proposed product contract based on the inspected
repository and recorded evidence. It is not yet a declaration that every
requirement is implemented.

Before implementation work begins: 1. review and approve the product
behavior and acceptance criteria; 2. retain the confirmed gaps as
explicit work items; 3. establish a test/evidence plan for each gap; 4.
make code changes only after the affected behavior and verification
criteria are agreed; 5. update this contract when behavior or public API
guarantees change.

**P0.1 completion condition:** Product scope, runtime boundaries,
truthful outcome semantics, acceptance thresholds, and known gaps are
documented and accepted. P0.1 completion does not mean that all gaps
above are resolved.

**Next phase:** P0.2 --- Architecture and Security Reconciliation, using
G-01 through G-06 as inputs and preserving the current
cognitive/execution separation.
