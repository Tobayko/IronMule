# Security policy

## Reporting a vulnerability

Report privately through GitHub:
[**open a security advisory**](https://github.com/Tobayko/IronMule/security/advisories/new).
Please do not open a public issue for a vulnerability.

This is a one-person project, so expect a first reply within a week rather than
within a day. There is no bounty programme.

Useful in a report: what an attacker can reach, the commands or requests that
reproduce it, and the versions of IronMule, Python and MLX you used.

## What is in scope

IronMule loads model weights, runs them, and serves them over HTTP. In scope:

- the HTTP server: authentication, the bind address, request parsing, anything
  that lets one caller read another caller's prompt, output or cache;
- the local state under `~/.ironmule`: profiles, tuning results and settings
  that a lower-privileged process should not be able to change;
- model and plan loading: a path, revision or profile that executes something it
  should not;
- the subprocess boundary used for measurement and for the reference worker.

## What is not in scope

- **What the model says.** Prompt injection, jailbreaks and wrong answers are
  properties of the model you loaded, not of this runtime.
- **A server you exposed yourself.** `ironmule serve` binds to `127.0.0.1` and
  expects a trusted local caller. Putting it on a public interface without a
  proxy, TLS and authentication is a deployment choice; see
  [docs/HTTP.md](docs/HTTP.md).
- **Model weights.** Nothing here redistributes them; trust in a checkpoint is
  between you and whoever published it.

## Supported versions

The latest release and `main`. Older tags get no fixes — this project is young
enough that upgrading is the fix.
