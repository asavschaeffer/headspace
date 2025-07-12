# Summary of Globule Wiki Reorganization and Naming Convention

This document outlines the process and decisions made during the reorganization and renaming of the `globule.wiki` directory, aiming for a more structured, semantically organized, and GitHub-wiki-friendly layout.

## 1. Initial State

The project wiki began with a flat structure containing various markdown files and a few top-level directories (`.git`, `LLD_research`, `LLD-modules`, `old`, `roleplays`).

## 2. Problem Statement

The user sought assistance in organizing the project wiki directory to improve clarity and navigability.

## 3. Iterative Organization Proposals

Several organizational structures were proposed and refined:

*   **Initial Proposal (Generic):** `Design/`, `Engineering/`, `Archive/` with sub-levels for HLD/LLD.
*   **Revised Proposal (Spiral-inspired):** `Vision/`, `Design/`, `Code/`, `Understanding/`, `Archive/`.
*   **Simplified Proposal (Form/Texture):** `Form/`, `Texture/`, `Archive/`.
*   **Final Adopted Proposal (Component-Oriented):** This structure was chosen to mirror the internal architecture of the Globule system itself, grouping documents by the specific component they describe.

    *   `00_Home.md` (at root)
    *   `1_Foundations/`
    *   `2_System_Architecture/`
    *   `3_Core_Components/`
        *   `Configuration_System/`
        *   `Schema_Engine/`
    *   `_Archive/`
        *   `old/`
        *   `roleplays/`

## 4. Key Decisions and Actions

*   **Creation of `home.md`:** A central `home.md` file was created at the root to serve as the main entry point and to document the directory structure.
*   **Comprehensive File Reading:** All existing markdown files were read to gain a deep understanding of the project's vision, design, and technical details. This understanding informed subsequent decisions.
*   **Naming Convention Rules:** Strict rules were established for file naming:
    1.  **Explicit Numbering:** Files are numbered (e.g., `10_`, `20_`) to ensure correct ordering in GitHub wikis, allowing for future insertions.
    2.  **Separator Usage:** `-` (hyphen) is used between related conceptual words (e.g., `High-Level-Design`), and `_` (underscore) is used to separate distinct ideas or metadata (e.g., `LLD_Configuration-System`).
    3.  **Clarity & No Redundancy:** Filenames are clear and self-describing without repeating information already conveyed by the directory structure (e.g., `LLD_` prefix for Low-Level Design documents, `Research_` for research documents).
*   **Execution of Reorganization:**
    *   The new directory structure was created.
    *   Files were moved into their respective new directories.
    *   Files were systematically renamed according to the agreed-upon naming conventions.
    *   Empty original directories (`LLD-modules`, `LLD_research`) were removed.
*   **GitHub Wiki Validation:** The user confirmed that the new numbering and naming scheme effectively organized the wiki on GitHub, despite the platform's flattening of folder structures.

## 5. Current State

The `globule.wiki` directory is now organized according to the component-oriented structure, with all files renamed to follow the established numbering and naming conventions. The `00_Home.md` file exists at the root, containing the initial directory map.
