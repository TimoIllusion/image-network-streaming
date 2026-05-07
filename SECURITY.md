# Security Policy

## Reporting a vulnerability

This project is a benchmarking harness intended for trusted local networks; it is not hardened for exposure to the public internet. See the **Network exposure** section in [README.md](README.md) for the deployment posture and the `INFSB_BIND` / `INFSB_TOKEN` knobs.

If you believe you have found a security issue, please report it privately to **tml9000@posteo.de** rather than opening a public GitHub issue. Include:

- a description of the issue and its impact
- a minimal reproduction (commands, env vars, transport in use)
- the package version (`python -c "import inference_streaming_benchmark as m; print(m.__version__)"`)

You can expect an acknowledgement within a few days. There is no formal SLA — this is a single-maintainer hobby project.

## Supported versions

Only the latest tagged release on `main` is supported. There are no backports.
