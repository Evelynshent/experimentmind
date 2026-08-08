# Clean-room implementation

ExperimentMind is an independent implementation built from the generic
requirements in SPEC.md.

Do not use, reproduce, infer, or import any employer source code,
datasets, prompts, schemas, SQL, internal documentation, test cases,
credentials, proprietary terminology, or internal implementation details.

All implementation decisions must be derived independently from SPEC.md
and public/open-source knowledge.

## V1 audit

- Project-specific behavior is derived only from `SPEC.md`.
- All experiment observations are fictional and generated locally from a fixed
  random seed.
- Statistical methods come from public SciPy and statsmodels APIs.
- Structured output uses the public OpenAI API and Pydantic schemas.
- The repository contains no employer code, datasets, prompts, schemas, SQL,
  credentials, internal documentation, or proprietary terminology.
- V1 adds no database, RAG system, memory, knowledge graph, agent orchestration,
  web application, or external data integration.
