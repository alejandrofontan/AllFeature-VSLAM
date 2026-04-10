---
name: clean-function
description: Clean and optimize a C++ function. Usage: /clean-function <src_file> <header_file> <function_name>
---

You are helping clean and optimize a specific C++ function. The user will provide:
- A source file (`.cpp` or `.cc`)
- A header file (`.h`)
- The name of the function to clean

## Steps

### 1. Read the files
Read both the source file and the header file in full.

### 2. Identify clean reference functions
Before touching the target function, scan the rest of the source file for functions that are already well-written. Use these as the style and quality reference for this codebase. Note patterns such as:
- Naming conventions (variables, parameters, types)
- Comment style and density
- Use of `const`, references, smart pointers
- Error handling patterns
- Code structure and formatting

### 3. Locate the target function
Find the target function in both the header (declaration) and source (definition). Read it carefully.

### 4. Clean the syntax
Rewrite the function applying the style conventions observed in step 2. Specifically:
- Fix inconsistent formatting (indentation, spacing, brace style)
- Normalize naming to match the codebase conventions
- Remove dead code, redundant comments, commented-out blocks, and debug prints
- Fix include ordering if relevant sections are touched
- Ensure `const`-correctness on parameters and member functions
- Replace raw loops with cleaner equivalents where it matches existing style

Present the cleaned version with a brief explanation of what changed syntactically.

### 5. Suggest optimizations
After the cleaned version, list concrete optimization suggestions focused on **speed** and **memory efficiency**. For each suggestion:
- Describe the change
- Explain *why* it helps (cache locality, fewer allocations, reduced copies, SIMD-friendliness, etc.)
- Show a code snippet if the change is non-trivial

Prioritize suggestions by expected impact. Do not implement them — only suggest, so the user can decide what to apply.
