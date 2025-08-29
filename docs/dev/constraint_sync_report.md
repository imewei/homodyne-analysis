# Parameter Constraints Synchronization Report

**Generated:** 2025-08-20 10:59:00

## Overview

This report documents the synchronization of parameter constraints and ranges across the
homodyne package to ensure consistency between documentation and configuration files.

## Authoritative Parameter Constraints

The following parameter constraints were established as the single source of truth:

### Core Model Parameters

- **D0**: [1.0, 1000000.0] Å²/s (TruncatedNormal)
- **alpha**: [-2.0, 2.0] dimensionless (Normal)
- **D_offset**: [-100, 100] Å²/s (Normal)
- **gamma_dot_t0**: [1e-06, 1.0] s⁻¹ (TruncatedNormal)
- **beta**: [-2.0, 2.0] dimensionless (Normal)
- **gamma_dot_t_offset**: [-0.01, 0.01] s⁻¹ (Normal)
- **phi0**: [-10, 10] degrees (Normal)

### Scaling Parameters

- **contrast**: [0.05, 0.5] (TruncatedNormal)
- **offset**: [0.05, 1.95] (TruncatedNormal)

## Changes Made

Total changes: 35

### general

- Backed up README.md to README.md.backup_20250820_105900
- Successfully wrote updated README.md
- Backed up homodyne_config.json to homodyne_config.json.backup_20250820_105900
- Successfully wrote updated homodyne_config.json
- Backed up my_config.json to my_config.json.backup_20250820_105900
- Successfully wrote updated my_config.json
- Backed up homodyne_config.json to homodyne_config.json.backup_20250820_105900
- Successfully wrote updated homodyne_config.json
- Backed up my_config.json to my_config.json.backup_20250820_105900
- Successfully wrote updated my_config.json
- Backed up homodyne/config_laminar_flow.json to
  homodyne/config_laminar_flow.json.backup_20250820_105900
- Successfully wrote updated homodyne/config_laminar_flow.json
- Backed up homodyne/config_template.json to
  homodyne/config_template.json.backup_20250820_105900
- Successfully wrote updated homodyne/config_template.json
- Backed up homodyne/config_static_isotropic.json to
  homodyne/config_static_isotropic.json.backup_20250820_105900
- Successfully wrote updated homodyne/config_static_isotropic.json
- Backed up homodyne/config_static_anisotropic.json to
  homodyne/config_static_anisotropic.json.backup_20250820_105900
- Successfully wrote updated homodyne/config_static_anisotropic.json

### README.md

- Updated parameter constraints section in README.md

### my_config.json

- Added reference bounds for fixed parameter gamma_dot_t0 in my_config.json
- Added reference bounds for fixed parameter beta in my_config.json
- Added reference bounds for fixed parameter gamma_dot_t_offset in my_config.json
- Added reference bounds for fixed parameter phi0 in my_config.json
- Added reference bounds for fixed parameter gamma_dot_t0 in my_config.json
- Added reference bounds for fixed parameter beta in my_config.json
- Added reference bounds for fixed parameter gamma_dot_t_offset in my_config.json
- Added reference bounds for fixed parameter phi0 in my_config.json

### homodyne/config_static_isotropic.json

- Added reference bounds for fixed parameter gamma_dot_t0 in
  homodyne/config_static_isotropic.json
- Added reference bounds for fixed parameter beta in
  homodyne/config_static_isotropic.json
- Added reference bounds for fixed parameter gamma_dot_t_offset in
  homodyne/config_static_isotropic.json
- Added reference bounds for fixed parameter phi0 in
  homodyne/config_static_isotropic.json

### homodyne/config_static_anisotropic.json

- Added reference bounds for fixed parameter gamma_dot_t0 in
  homodyne/config_static_anisotropic.json
- Added reference bounds for fixed parameter beta in
  homodyne/config_static_anisotropic.json
- Added reference bounds for fixed parameter gamma_dot_t_offset in
  homodyne/config_static_anisotropic.json
- Added reference bounds for fixed parameter phi0 in
  homodyne/config_static_anisotropic.json

## Validation

After applying these changes, all parameter constraints should be consistent across:

- Main README.md documentation
- All configuration template files
- All configuration example files

## Next Steps

1. Review the backup files created during this process
1. Test the updated configurations
1. Commit the changes to version control
1. Update the changelog

## Files Modified

- README.md
- homodyne/config_laminar_flow.json
- homodyne/config_static_anisotropic.json
- homodyne/config_static_isotropic.json
- homodyne/config_template.json
- homodyne_config.json
- my_config.json
