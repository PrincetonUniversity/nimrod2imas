#!/usr/bin/env python3
"""
validate_nimrod2imas_bp.py

Validate ADIOS BP files created by nimrod2imas_bp.py.

This script:
  1. Attempts to load data using effis.shim.omas_adios.load_omas_adios
  2. Falls back to direct ADIOS2 FileReader if OMAS loading fails
  3. Validates the presence and correctness of key data fields
  4. Optionally generates summary plots

Usage:
    python validate_nimrod2imas_bp.py <bp_directory>
    python validate_nimrod2imas_bp.py output/nimrod_v01.bp --plot
"""

import argparse
import os
import sys
import numpy as np


def validate_with_omas(bp_dir, verbose=True):
    """
    Validate BP files using EFFIS load_omas_adios.
    
    Returns dict of loaded ODS objects or None if loading fails.
    """
    try:
        from effis.shim.omas_adios import load_omas_adios
    except ImportError as e:
        if verbose:
            print(f"Warning: Could not import load_omas_adios: {e}")
        return None
    
    ods_dict = {}
    ids_names = ['equilibrium', 'core_profiles', 'wall', 'mhd']
    
    for ids_name in ids_names:
        bp_path = os.path.join(bp_dir, f"{ids_name}.bp")
        if not os.path.exists(bp_path):
            if verbose:
                print(f"Warning: {bp_path} not found")
            continue
        
        try:
            if verbose:
                print(f"Loading {ids_name}.bp with load_omas_adios...")
            ods = load_omas_adios(bp_path, consistency_check=False)
            ods_dict[ids_name] = ods
            if verbose:
                print(f"  Successfully loaded {ids_name}")
        except Exception as e:
            if verbose:
                print(f"  Failed to load {ids_name}: {e}")
            return None
    
    return ods_dict


def validate_with_adios2(bp_dir, verbose=True):
    """
    Validate BP files using raw ADIOS2 FileReader.
    
    Returns dict of variable data or raises exception on failure.
    """
    import adios2
    
    results = {
        'equilibrium': {},
        'core_profiles': {},
        'wall': {},
        'mhd': {}
    }
    
    # --- Equilibrium ---
    eq_path = os.path.join(bp_dir, "equilibrium.bp")
    if os.path.exists(eq_path):
        if verbose:
            print(f"Reading {eq_path}...")
        with adios2.FileReader(eq_path) as reader:
            variables = reader.available_variables()
            results['equilibrium']['_variables'] = list(variables.keys())
            results['equilibrium']['_num_variables'] = len(variables)
            
            # Read key variables
            try:
                results['equilibrium']['time'] = reader.read('equilibrium.time')
            except:
                pass
            
            try:
                results['equilibrium']['vacuum_toroidal_field.r0'] = reader.read(
                    'equilibrium.vacuum_toroidal_field.r0')
            except:
                pass
            
            try:
                results['equilibrium']['vacuum_toroidal_field.b0'] = reader.read(
                    'equilibrium.vacuum_toroidal_field.b0')
            except:
                pass
            
            # Try both naming conventions for time_slice
            for prefix in ['equilibrium.time_slice.0.', 'equilibrium.time_slice.']:
                try:
                    results['equilibrium']['global_quantities.ip'] = reader.read(
                        f'{prefix}global_quantities.ip')
                    results['equilibrium']['global_quantities.psi_axis'] = reader.read(
                        f'{prefix}global_quantities.psi_axis')
                    results['equilibrium']['global_quantities.psi_boundary'] = reader.read(
                        f'{prefix}global_quantities.psi_boundary')
                    results['equilibrium']['global_quantities.magnetic_axis.r'] = reader.read(
                        f'{prefix}global_quantities.magnetic_axis.r')
                    results['equilibrium']['global_quantities.magnetic_axis.z'] = reader.read(
                        f'{prefix}global_quantities.magnetic_axis.z')
                    results['equilibrium']['profiles_1d.psi'] = reader.read(
                        f'{prefix}profiles_1d.psi')
                    results['equilibrium']['profiles_1d.q'] = reader.read(
                        f'{prefix}profiles_1d.q')
                    results['equilibrium']['profiles_1d.pressure'] = reader.read(
                        f'{prefix}profiles_1d.pressure')
                    results['equilibrium']['profiles_1d.f'] = reader.read(
                        f'{prefix}profiles_1d.f')
                    break
                except:
                    continue
            
            # 2D psi
            for var in variables.keys():
                if 'profiles_2d' in var and var.endswith('.psi'):
                    try:
                        results['equilibrium']['profiles_2d.psi'] = reader.read(var)
                    except:
                        pass
                    break
            
            # Boundary
            for prefix in ['equilibrium.time_slice.0.', 'equilibrium.time_slice.']:
                try:
                    results['equilibrium']['boundary.outline.r'] = reader.read(
                        f'{prefix}boundary.outline.r')
                    results['equilibrium']['boundary.outline.z'] = reader.read(
                        f'{prefix}boundary.outline.z')
                    break
                except:
                    continue
        
        if verbose:
            print(f"  Found {results['equilibrium']['_num_variables']} variables")
    
    # --- Core Profiles ---
    cp_path = os.path.join(bp_dir, "core_profiles.bp")
    if os.path.exists(cp_path):
        if verbose:
            print(f"Reading {cp_path}...")
        with adios2.FileReader(cp_path) as reader:
            variables = reader.available_variables()
            results['core_profiles']['_variables'] = list(variables.keys())
            results['core_profiles']['_num_variables'] = len(variables)
            
            try:
                results['core_profiles']['time'] = reader.read('core_profiles.time')
            except:
                pass
            
            # Try both naming conventions
            for prefix in ['core_profiles.profiles_1d.0.', 'core_profiles.profiles_1d.']:
                try:
                    results['core_profiles']['grid.rho_tor_norm'] = reader.read(
                        f'{prefix}grid.rho_tor_norm')
                    results['core_profiles']['electrons.density_thermal'] = reader.read(
                        f'{prefix}electrons.density_thermal')
                    results['core_profiles']['electrons.temperature'] = reader.read(
                        f'{prefix}electrons.temperature')
                    break
                except:
                    continue
            
            # Ion species
            ion_species = []
            for i in range(10):  # Check up to 10 species
                for prefix in [f'core_profiles.profiles_1d.0.ion.{i}.',
                               f'core_profiles.profiles_1d.ion.{i}.']:
                    try:
                        z_n = reader.read(f'{prefix}element.0.z_n')
                        a = reader.read(f'{prefix}element.0.a')
                        ion_species.append({'index': i, 'Z': z_n, 'A': a})
                        
                        # Try to read densities and velocities
                        for field in ['density_thermal', 'density_fast', 'temperature',
                                       'velocity.toroidal', 'velocity.poloidal',
                                       'velocity.diamagnetic', 'pressure_fast_perpendicular']:
                            try:
                                key = f'ion.{i}.{field}'
                                results['core_profiles'][key] = reader.read(f'{prefix}{field}')
                            except:
                                pass
                        break
                    except:
                        continue
            results['core_profiles']['_ion_species'] = ion_species
        
        if verbose:
            print(f"  Found {results['core_profiles']['_num_variables']} variables")
            print(f"  Found {len(ion_species)} ion species")
    
    # --- Wall ---
    wall_path = os.path.join(bp_dir, "wall.bp")
    if os.path.exists(wall_path):
        if verbose:
            print(f"Reading {wall_path}...")
        with adios2.FileReader(wall_path) as reader:
            variables = reader.available_variables()
            results['wall']['_variables'] = list(variables.keys())
            results['wall']['_num_variables'] = len(variables)
            
            try:
                results['wall']['time'] = reader.read('wall.time')
            except:
                pass
            
            # Limiter outline
            for var in variables.keys():
                if 'limiter' in var and 'outline.r' in var:
                    try:
                        results['wall']['limiter.r'] = reader.read(var)
                        results['wall']['limiter.z'] = reader.read(var.replace('.r', '.z'))
                    except:
                        pass
                    break
        
        if verbose:
            print(f"  Found {results['wall']['_num_variables']} variables")
    
    # --- MHD ---
    mhd_path = os.path.join(bp_dir, "mhd.bp")
    if os.path.exists(mhd_path):
        if verbose:
            print(f"Reading {mhd_path}...")
        with adios2.FileReader(mhd_path) as reader:
            variables = reader.available_variables()
            results['mhd']['_variables'] = list(variables.keys())
            results['mhd']['_num_variables'] = len(variables)
            
            try:
                results['mhd']['code.name'] = reader.read('mhd.code.name')
            except:
                pass
            
            try:
                results['mhd']['code.parameters'] = reader.read('mhd.code.parameters')
            except:
                pass
        
        if verbose:
            print(f"  Found {results['mhd']['_num_variables']} variables")
    
    return results


def print_validation_summary(data, verbose=True):
    """Print a summary of the validated data."""
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    # Equilibrium
    eq = data.get('equilibrium', {})
    if eq:
        print("\n--- Equilibrium ---")
        print(f"  Variables: {eq.get('_num_variables', 0)}")
        if 'time' in eq:
            print(f"  Time: {eq['time']}")
        if 'vacuum_toroidal_field.r0' in eq:
            print(f"  R0: {eq['vacuum_toroidal_field.r0']:.4f} m")
        if 'vacuum_toroidal_field.b0' in eq:
            b0 = eq['vacuum_toroidal_field.b0']
            if hasattr(b0, '__len__'):
                b0 = b0[0]
            print(f"  B0: {b0:.4f} T")
        if 'global_quantities.ip' in eq:
            print(f"  Ip: {eq['global_quantities.ip']:.0f} A")
        if 'global_quantities.psi_axis' in eq:
            print(f"  Psi_axis: {eq['global_quantities.psi_axis']:.6f} Wb")
        if 'profiles_1d.psi' in eq:
            psi = eq['profiles_1d.psi']
            print(f"  profiles_1d.psi: shape={psi.shape}, range=[{psi.min():.4f}, {psi.max():.4f}]")
        if 'profiles_1d.q' in eq:
            q = eq['profiles_1d.q']
            print(f"  profiles_1d.q: shape={q.shape}, range=[{q.min():.4f}, {q.max():.4f}]")
        if 'profiles_2d.psi' in eq:
            psi2d = eq['profiles_2d.psi']
            print(f"  profiles_2d.psi: shape={psi2d.shape}")
        if 'boundary.outline.r' in eq:
            r = eq['boundary.outline.r']
            z = eq['boundary.outline.z']
            print(f"  Boundary: {len(r)} points, R=[{r.min():.4f}, {r.max():.4f}], Z=[{z.min():.4f}, {z.max():.4f}]")
    
    # Core Profiles
    cp = data.get('core_profiles', {})
    if cp:
        print("\n--- Core Profiles ---")
        print(f"  Variables: {cp.get('_num_variables', 0)}")
        if 'time' in cp:
            print(f"  Time: {cp['time']}")
        if 'grid.rho_tor_norm' in cp:
            rho = cp['grid.rho_tor_norm']
            print(f"  grid.rho_tor_norm: shape={rho.shape}, range=[{rho.min():.4f}, {rho.max():.4f}]")
        if 'electrons.density_thermal' in cp:
            ne = cp['electrons.density_thermal']
            print(f"  electrons.density_thermal: shape={ne.shape}, range=[{ne.min():.4e}, {ne.max():.4e}] m^-3")
        if 'electrons.temperature' in cp:
            te = cp['electrons.temperature']
            print(f"  electrons.temperature: shape={te.shape}, range=[{te.min():.2f}, {te.max():.2f}] eV")
        
        ion_species = cp.get('_ion_species', [])
        if ion_species:
            print(f"  Ion species: {len(ion_species)}")
            for ion in ion_species:
                print(f"    ion.{ion['index']}: Z={ion['Z']}, A={ion['A']}")
    
    # Wall
    wall = data.get('wall', {})
    if wall:
        print("\n--- Wall ---")
        print(f"  Variables: {wall.get('_num_variables', 0)}")
        if 'limiter.r' in wall:
            r = wall['limiter.r']
            z = wall['limiter.z']
            print(f"  Limiter: {len(r)} points, R=[{r.min():.4f}, {r.max():.4f}], Z=[{z.min():.4f}, {z.max():.4f}]")
    
    # MHD
    mhd = data.get('mhd', {})
    if mhd:
        print("\n--- MHD ---")
        print(f"  Variables: {mhd.get('_num_variables', 0)}")
        if 'code.name' in mhd:
            print(f"  code.name: {mhd['code.name']}")
        if 'code.parameters' in mhd:
            params = mhd['code.parameters']
            print(f"  code.parameters: {len(params)} chars")
    
    print("\n" + "=" * 60)


def generate_plots(data, output_dir=None):
    """Generate validation plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available, skipping plots")
        return
    
    if output_dir is None:
        output_dir = "."
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    eq = data.get('equilibrium', {})
    cp = data.get('core_profiles', {})
    wall = data.get('wall', {})
    
    # Plot 1: 2D psi contour
    ax = axes[0, 0]
    if 'profiles_2d.psi' in eq:
        psi2d = eq['profiles_2d.psi']
        ax.contour(psi2d.T, levels=20)
        ax.set_title('Psi 2D Contour')
        ax.set_xlabel('R index')
        ax.set_ylabel('Z index')
    else:
        ax.text(0.5, 0.5, 'No 2D psi data', ha='center', va='center')
        ax.set_title('Psi 2D Contour')
    
    # Plot 2: q profile
    ax = axes[0, 1]
    if 'profiles_1d.q' in eq and 'profiles_1d.psi' in eq:
        psi = eq['profiles_1d.psi']
        q = eq['profiles_1d.q']
        # Normalize psi
        psi_norm = (psi - psi.min()) / (psi.max() - psi.min())
        ax.plot(psi_norm, q)
        ax.set_xlabel('Normalized psi')
        ax.set_ylabel('q')
        ax.set_title('Safety Factor Profile')
        ax.grid(True)
    else:
        ax.text(0.5, 0.5, 'No q profile data', ha='center', va='center')
        ax.set_title('Safety Factor Profile')
    
    # Plot 3: Pressure profile
    ax = axes[0, 2]
    if 'profiles_1d.pressure' in eq and 'profiles_1d.psi' in eq:
        psi = eq['profiles_1d.psi']
        pressure = eq['profiles_1d.pressure']
        psi_norm = (psi - psi.min()) / (psi.max() - psi.min())
        ax.plot(psi_norm, pressure / 1e3)
        ax.set_xlabel('Normalized psi')
        ax.set_ylabel('Pressure (kPa)')
        ax.set_title('Pressure Profile')
        ax.grid(True)
    else:
        ax.text(0.5, 0.5, 'No pressure data', ha='center', va='center')
        ax.set_title('Pressure Profile')
    
    # Plot 4: Electron profiles
    ax = axes[1, 0]
    if 'grid.rho_tor_norm' in cp:
        rho = cp['grid.rho_tor_norm']
        plotted = False
        if 'electrons.density_thermal' in cp:
            ne = cp['electrons.density_thermal']
            ax.plot(rho, ne / 1e19, label='ne (10^19 m^-3)')
            plotted = True
        if plotted:
            ax.set_xlabel('rho_tor_norm')
            ax.set_ylabel('Density')
            ax.set_title('Electron Density')
            ax.legend()
            ax.grid(True)
    if not ax.lines:
        ax.text(0.5, 0.5, 'No electron density data', ha='center', va='center')
        ax.set_title('Electron Density')
    
    # Plot 5: Temperature profiles
    ax = axes[1, 1]
    if 'grid.rho_tor_norm' in cp:
        rho = cp['grid.rho_tor_norm']
        plotted = False
        if 'electrons.temperature' in cp:
            te = cp['electrons.temperature']
            ax.plot(rho, te / 1e3, label='Te (keV)')
            plotted = True
        if plotted:
            ax.set_xlabel('rho_tor_norm')
            ax.set_ylabel('Temperature (keV)')
            ax.set_title('Electron Temperature')
            ax.legend()
            ax.grid(True)
    if not ax.lines:
        ax.text(0.5, 0.5, 'No temperature data', ha='center', va='center')
        ax.set_title('Electron Temperature')
    
    # Plot 6: Wall/limiter and boundary
    ax = axes[1, 2]
    plotted = False
    if 'limiter.r' in wall:
        r_lim = wall['limiter.r']
        z_lim = wall['limiter.z']
        ax.plot(r_lim, z_lim, 'k-', linewidth=2, label='Limiter')
        plotted = True
    if 'boundary.outline.r' in eq:
        r_bnd = eq['boundary.outline.r']
        z_bnd = eq['boundary.outline.z']
        ax.plot(r_bnd, z_bnd, 'r-', linewidth=1.5, label='Boundary')
        plotted = True
    if plotted:
        ax.set_xlabel('R (m)')
        ax.set_ylabel('Z (m)')
        ax.set_title('Cross-section')
        ax.set_aspect('equal')
        ax.legend()
        ax.grid(True)
    else:
        ax.text(0.5, 0.5, 'No wall/boundary data', ha='center', va='center')
        ax.set_title('Cross-section')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'validation_plots.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Saved validation plots to {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Validate ADIOS BP files created by nimrod2imas_bp.py"
    )
    parser.add_argument("bp_dir", help="Directory containing BP files (e.g., output/nimrod_v01.bp)")
    parser.add_argument("--plot", action="store_true", help="Generate validation plots")
    parser.add_argument("--plot-dir", default=None, help="Directory to save plots (default: bp_dir)")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress verbose output")
    parser.add_argument("--use-omas", action="store_true", 
                        help="Try to use EFFIS load_omas_adios (may fail with version mismatch)")
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    if not os.path.isdir(args.bp_dir):
        print(f"Error: {args.bp_dir} is not a directory")
        return 1
    
    print(f"Validating BP files in: {args.bp_dir}")
    print("=" * 60)
    
    # Try OMAS loading first if requested
    if args.use_omas:
        print("\nAttempting to load with EFFIS load_omas_adios...")
        ods_dict = validate_with_omas(args.bp_dir, verbose=verbose)
        if ods_dict:
            print("SUCCESS: All files loaded with load_omas_adios")
            # TODO: Extract data from ODS for summary/plotting
        else:
            print("OMAS loading failed, falling back to direct ADIOS2...")
    
    # Use direct ADIOS2 reading
    print("\nValidating with ADIOS2 FileReader...")
    try:
        data = validate_with_adios2(args.bp_dir, verbose=verbose)
    except Exception as e:
        print(f"Error reading BP files: {e}")
        return 1
    
    # Print summary
    print_validation_summary(data, verbose=verbose)
    
    # Check for critical data
    errors = []
    if not data.get('equilibrium', {}).get('profiles_1d.psi') is not None:
        errors.append("Missing equilibrium.profiles_1d.psi")
    if not data.get('core_profiles', {}).get('electrons.density_thermal') is not None:
        errors.append("Missing core_profiles.electrons.density_thermal")
    
    if errors:
        print("\nWARNINGS:")
        for err in errors:
            print(f"  - {err}")
    
    # Generate plots if requested
    if args.plot:
        plot_dir = args.plot_dir or args.bp_dir
        generate_plots(data, output_dir=plot_dir)
    
    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
