# GCOPTER provenance and bootstrap contract

- Upstream clean-reproduction URL: https://github.com/canimiliya/GCOPTER-Clean-Reproduction.git
- Frozen clean-reproduction ref used by the prior audit: `7e7fdd8258bb345dd868cbc684f9e947ac2f34e4`
- Official GCOPTER source commit: `e0444f6d47b84f972ced91746b05feb36ce1fd4f`
- License: MIT (`LICENSE` in the clean reproduction checkout).
- Local checkout is never assumed. Set `GCOPTER_CLEAN_REPRODUCTION` to the
  checkout root, or let `scripts/bootstrap_windows.ps1` create
  `.deps/GCOPTER-Clean-Reproduction`.
- Required external inputs are the nine read-only files under
  `scenes/pybullet/*.yaml`. Full expert-data generation additionally needs the
  compiled `gcopter_yaml_scene_planner` executable, configured with
  `GCOPTER_YAML_SCENE_PLANNER` or supplied at `.deps/gcopter_reference/`.
- Runtime consumption is limited to the analytic position/velocity reference
  and frozen scene geometry. Motor RPM, DSLPID output, optimizer state,
  corridor coefficients, and planner cost are not student observations.
- The Git repository does not copy a GCOPTER source mirror. Existing `.deps/`
  material is ignored local input and must be recreated or restored separately.
