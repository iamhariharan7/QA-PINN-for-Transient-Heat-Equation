# Contributing to the Framework

Contributions are welcome. Please ensure that the core physics logic and dataset generators are not broken. Run pytest before submitting a PR.

## Development Workflow
1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Ensure pytest passes fully
5. Submit a Pull Request

## Code Style Rules
- Follow PEP 8 guidelines
- Document functions using Google style docstrings
- Maintain strict typing where applicable

## Testing Requirements
All new features must be accompanied by unit tests placed in the 	ests/ directory.

## Pull Request Process
Use the provided Pull Request template and ensure all checklist items are completed before requesting a review.

## CHANGE CONTROL POLICY
Before making any modification to the existing project:
1. Analyze the current project structure completely.
2. Identify affected files and dependencies.
3. Explain the purpose of every proposed change.
4. Only implement non-breaking additions.
5. Maintain complete backward compatibility.
6. Preserve the existing execution workflow.
7. Update CHANGELOG.md for every structural or functional addition.
8. Keep modifications isolated and easy to revert.
9. Create backups before modifying existing files.
10. Never perform large-scale refactoring automatically.

The AI (or human contributor) must behave as a software maintainer, not as a code rewrite engine.
If a requested improvement requires changing existing working logic: **STOP.**
Create a recommendation document explaining: Why the change is needed, Files affected, Possible risks, Migration steps.
Do not directly modify working implementation without approval.

## BACKWARD COMPATIBILITY REQUIREMENT

Before adding any new GitHub, documentation, or automation files:
1. Analyze the existing repository structure first.
2. Preserve all existing Folder names, File paths, Module names, Import paths, Configuration files, Data formats, Output formats, and Existing workflows.
3. Do NOT Rename existing folders, Move existing files, Refactor existing Python code, Modify solver logic, Modify dataset generation logic, Modify model training logic, Change APIs or function signatures, Remove existing functionality, or Replace existing implementations.
4. Only create additional files required for GitHub management, Documentation, Continuous integration, Contribution guidelines, and Project metadata.
5. All additions must be isolated and must not affect the current execution flow.
6. After adding new repository files, verify that Existing scripts run without modification, Existing imports remain valid, Existing dataset generation works, Existing output generation works, and Existing experiments remain reproducible.
The project must remain fully backward compatible before and after GitHub repository improvements.

## REPOSITORY MODIFICATION SAFETY REQUIREMENT

Before adding or modifying any repository-level GitHub, documentation, or automation files:
1. Analyze the existing repository structure first.
2. Preserve all existing Folder names, File paths, Module names, Import paths, Configuration files, Data formats, Output formats, and Existing workflows.
3. Do NOT Rename existing folders, Move existing files, Refactor existing Python code, Modify solver logic, Modify dataset generation logic, Modify model training logic, Change APIs or function signatures, Remove existing functionality, or Replace existing implementations.
4. Only add or modify files required for GitHub repository management, Documentation, Continuous integration, Contribution guidelines, and Project metadata.
5. All repository-level improvements must be isolated and must not affect the current execution flow.
6. After any repository-level changes, verify that Existing scripts execute successfully, Existing imports remain valid, Existing dataset generation works correctly, Existing output generation works correctly, and Existing experiments remain reproducible.
The project must remain fully backward compatible before and after all repository improvements.
