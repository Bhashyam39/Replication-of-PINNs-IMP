# Plan: Guide implementation of the uploaded PINNs paper

## Stage 0 — Scope clarification
- Confirm target scope: 1D solidification replication, 3D NIST PINN, or both.
- Confirm framework preference and whether FEM/NIST data are available.
- Default recommendation if no constraint is given: implement Stage 0+1 from the guide first — analytical 1D baseline plus data-free 1D PINN with hard/soft Dirichlet BC comparison.

## Stage 1 — Coding workflow setup
- Load `/app/.agents/skills/vibecoding-general-swarm/SKILL.md` because this is a non-web coding implementation task.
- Follow its orchestration requirements for implementation, validation, and delivery.

## Stage 2 — Implementation
- Build a clean, runnable starter package under `/mnt/agents/output/` for the confirmed scope.
- Minimum default deliverable: analytical 1D solidification baseline + PINN implementation with corrected Swish and corrected hard-BC Heaviside ramp, training script, evaluation script, plots/metrics, and README with run commands.
- Mark all paper-missing hyperparameters as configurable assumptions.

## Stage 3 — Validation
- Run smoke tests and, where feasible, short training/evaluation to verify the code executes.
- Check analytical interface position at 10 s ≈ 22.44 mm.
- Verify hard BC enforces boundary values exactly by construction and soft BC is implemented as a loss term.
- Document any runtime or dependency limitations.

## Stage 4 — Delivery
- Provide the code path and a concise walkthrough of files, commands, and expected outputs.
- If the user chooses the 3D NIST scope, clearly separate what is implemented from what still requires external FEM/NIST data.
