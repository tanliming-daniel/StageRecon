# StageRecon

Parent repository for TV show stage reconstruction experiments.

This repository tracks the upstream projects as Git submodules:

- `ShowMak3r`: compositional TV show reconstruction pipeline.
- `sitcoms3D`: sitcom scene data, cameras, COLMAP reconstructions, and utilities.

Clone with submodules:

```bash
git clone --recurse-submodules <repo-url>
```

Initialize submodules after cloning without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```
