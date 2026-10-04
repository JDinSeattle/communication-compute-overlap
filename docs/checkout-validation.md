# Checkout validation

This local validation is separate from the user-confirmed cloud results in the README.

20 tests passed outside the sandbox with real loopback sockets; the sandbox-only attempt failed due to denied socket creation.

## Commands

```text
python3 -m unittest discover -s tests -v
```

```text
python3 -m unittest discover -s tests -v (outside sandbox; real loopback sockets)
```

No new physical-GPU performance, cloud rerun, or production availability claim is made. Local build and runtime artifacts remain outside Git; the committed source and profiles reproduce the checks with the pinned dependencies. Historical source changes already present in the user’s working tree are preserved in this update.
