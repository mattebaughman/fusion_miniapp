# Environment

Built 2026-11-18 on a laptop (macOS, Apple silicon, arm64) with uv 0.11.26 and uv-managed CPython 3.12.13.
Nothing here has been built or run on Perlmutter, Stellar-AI or Lux.

## Rebuild

From this folder:

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[stack,academy,dev]"
```

`UV_CACHE_DIR` may point anywhere; it was a scratch folder for this build.
FlowGentic 0.5.0 is not on PyPI or tagged yet, so the `stack` extra installs it from GitHub at main commit
`f475907cd1cb324e04c0a7e33018f38e96204b74` (the commit that carries 0.5.0), with its extras `langgraph,rhapsody,mcp`.
The extras: `stack` = the PDF's stack (LangGraph, langchain-openai, RHAPSODY, FlowGentic); `academy` = academy-py; `dev` = pytest.
Both `stack` and `academy` also carry cloudpickle, which RHAPSODY needs to run `evaluate` in worker processes
(added at integration on 2026-11-19 with `uv pip install cloudpickle`, version 3.1.3).
The default install (no extras) is the simulation only: numpy, scipy, pyyaml, netCDF4, torch.

To reproduce this exact set, write the list below to a file and add `-r <file>` to the install line.

## Installed versions (`uv pip freeze`, without the editable fusion_miniapp 0.1.0 itself)

```text
academy-py==1.0.0
aiohappyeyeballs==2.7.1
aiohttp==3.14.5
aiohttp-sse==3.0
aiologic==0.17.1
aiosignal==1.4.0
annotated-types==0.8.0
anyio==4.16.0
attrs==26.1.0
cachetools==8.0.1
certifi==2026.10.23
cffi==2.1.1
cftime==1.6.6.1
charset-normalizer==3.5.2
click==8.5.1
cloudpickle==3.1.3
cryptography==50.0.2
culsans==0.11.0
distro==1.9.0
filelock==4.2.12
flowgentic @ git+https://github.com/stride-research/flowgentic@f475907cd1cb324e04c0a7e33018f38e96204b74
frozenlist==1.8.0
fsspec==2026.9.0
globus-sdk==4.10.0
h11==0.16.0
httpcore==1.0.9
httpcore2==2.16.2
httpx==0.28.1
httpx2==2.16.2
idna==3.20
iniconfig==2.3.1
jinja2==3.1.6
jiter==0.19.0
jsonpatch==1.47
jsonpointer==3.12
jsonschema==4.26.0
jsonschema-specifications==2025.9.1
langchain==1.5.1
langchain-core==1.8.3
langchain-ollama==1.1.0
langchain-openai==1.8.2
langchain-protocol==0.0.21
langgraph==1.2.22
langgraph-checkpoint==4.3.2
langgraph-prebuilt==1.1.2
langgraph-sdk==0.4.14
langsmith==0.15.0
markupsafe==3.0.4
mcp==2.3.0
mcp-types==2.3.0
mpmath==1.3.0
multidict==7.2.1
netcdf4==1.7.5.1.post1
networkx==3.7
numpy==2.5.5
ollama==0.6.3
openai==3.55.1
opentelemetry-api==1.45.2
orjson==3.13.0
ormsgpack==1.13.1
packaging==26.3
pluggy==1.7.0
propcache==0.5.4
pycparser==3.12
pydantic==2.14.3
pydantic-core==2.50.4
pygments==2.21.0
pyjwt==2.15.1
pytest==9.2.0
python-multipart==0.0.34
pyyaml==6.0.3
redis==8.1.0
referencing==0.37.0
regex==2026.11.19
requests==2.34.2
requests-toolbelt==1.0.0
rhapsody-py==0.7.0
rpds-py==2026.9.1
scipy==1.18.1
setuptools==84.2.1
sniffio==1.3.1
sse-starlette==3.7.0
starlette==1.10.0
sympy==1.14.0
tenacity==9.2.1
tiktoken==0.14.0
tomli-w==1.2.0
torch==2.15.0
truststore==0.10.5
typeguard==4.6.0
typing-extensions==4.16.0
typing-inspection==0.4.5
urllib3==2.8.0
uuid-utils==0.19.0
uvicorn==0.54.0
websockets==17.4
wrapt==2.5.2
xxhash==4.1.0
yarl==1.25.1
zstandard==0.25.0
```

## What was checked on the laptop

- Imports: torch 2.15.0, netCDF4, numpy, scipy, yaml, academy, langgraph, langchain_openai, rhapsody, flowgentic, mcp, pytest.
  `torch.cuda.is_available()` is False, `torch.backends.mps.is_available()` is True; `torch.version.cuda` and `torch.version.hip` are None (a CPU/MPS build).
- The FlowGentic, RHAPSODY and Academy names the prototypes use import from these versions.
- Importing `fusion_miniapp` does not import torch.
- The inner-loop prototype (`inject_probe.py`) ran on MPS with this environment, both against the parent export and against `src/fusion_miniapp/parent/`.

## Not checked

CUDA, ROCm, Dragon (`dragonhpc`), RaDEX, vLLM or any real model server. On the three machines torch comes from
a CUDA or ROCm wheel index, so its build and version must be chosen per machine (named in the header of each
launch/*.sbatch; untested on the machines).
