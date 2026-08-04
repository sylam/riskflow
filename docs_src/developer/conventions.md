# Conventions

House rules for the riskflow codebase. These are enforced by convention, not by lint.

## JSON is the contract {#json-is-the-contract}

The job JSON is the whole program. End-user scripts do `import riskflow as rf`, `cx.load_json(...)`, `cx.run_job(...)` — **no internal imports, no monkey-patching**. Every framework feature ships behind a JSON switch, defaulting to bit-identical-when-off. The JSON format is documented in [JSON Configuration](../json/index.md) and generated from `fields.mapping`.

Consequences:

- **No defensive checks.** The JSON is validated by being the contract; code assumes required fields are present and fails loud (`KeyError` naming the factor via `check_tuple_name`) when they are not. Do not add `if x in d` guards for contract fields.
- **Config per variant, not flags.** A variant is a separate JSON config; scripts stay generic. Do not thread experiment booleans through function signatures.
- **Never monkey-patch internals.** A missing metric or behavior is added to the framework behind a switch — never reached into from a user script.
- **MarketData files are data-only** — no `Description` / commentary fields.

## `globals()` dispatch {#globals-dispatch}

Constructors dispatch on class name through the module `globals()`: `construct_calculation` does `globals().get(calc_type)(...)`; processes, deals, factors, and calibration classes are all resolved the same way from their JSON type string. This is why **one class per concept in a file** is the norm — the class name *is* the dispatch key. Adding a type means adding a class, not editing a dispatcher.

## Registries, not functions

Extension points are data. Factor discovery is three dicts (`dependant_fields`, `nested_fields`, `conditional_fields` — see [Dependency System](dependency_system.md)); process→factor wiring and calibration are registered in `Model Configuration` and `calibration_config.json`; JSON schema lives in `fields.mapping`. A function that switches on a magic-string type, or a parallel dict passed alongside a primary operand, is a class-or-registry waiting to happen — flag it. Strengthen the existing primitive; do not bolt on a parallel concept or a magic-string branch.

## Documentation and doc generation {#documentation-and-doc-generation}

Developer-facing model/deal/process docs are **class attributes**, harvested at build time by `riskflow_docs.py`:

- `documentation` (a `(section_name, [md_lines])` tuple) on classes in `stochasticprocess.py` / `instruments.py` / `bootstrappers.py` / `calculation.py` auto-publishes the Theory / Valuation / Bootstrapping / API pages. It is read **own-attr-only** (`cls.__dict__`, not MRO) so an alias subclass does not re-emit the parent's page.
- `fields.mapping` + `field_desc` drive the entire JSON reference tree (`generate_json_docs`).

So: model math and JSON field docs live **on the class / in `fields.py`**, not in prose pages. This developer section links to those generated pages; it does not restate them. The doc build (`ConstructMarkdown.build`) is described in the section README.

## Comment and code style {#code-style}

- Terse. Correct **>** efficient **>** least-lines, in that order — but no redundant work in hot paths (no stray `.to()` / `.item()` / `.detach()` in the AAD path).
- One-line comments explaining *why*, not *what*. No banner comments.
- Diagnostics at `logging.info`, never behind `os.environ` flags.
- A diff that removes lines/imports usually wins: small public surface, one-way dependency edges.

## Look before you write {#look-before-you-write}

Before adding any helper, **search `utils.py` and the package for an existing equivalent** — this is a hard rule, and every new-code task must do it first. Dedupe on contact when moving code. The name-resolution, curve-gather, date, and tensor primitives you need almost always already exist; a new one is usually a missed search.

## Change scope {#change-scope}

`credit_monte_carlo` and `base_valuation` are **do-not-touch** — the CVA/FVA/CollVA/IM block inside `Credit_Monte_Carlo.execute` in particular. They are the production valuation paths and everything downstream reconciles against the numbers they report. The `HedgeMonteCarlo` / solver stack is free to redesign. Reuse pre-processed deal data (`field_index['Cashflows']` etc.) inside `Deal` methods rather than re-walking `self.field`.

The one thing that *is* wired into that block is [boundary-correction assembly](calc_lifecycle.md#boundary-corrections-the-sensitivity-subsystem), and the terms on which it got there are the terms for anything else: it changes what is handed to `backward()` and **nothing** about what is reported. The correction is worth exactly zero in the forward pass by construction, and that is gated bit-identically — `np.array_equal` on the exposure and `==` on the scalar, with sensitivities on versus off. A change that cannot make that guarantee does not belong here.

## No overengineering

One-line fixes — a default, a JSON field, a file move — before a helper module. Do not introduce abstraction ahead of a second caller.
