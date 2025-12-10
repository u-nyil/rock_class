# Jr Calculator - Per specific plane
# Based on Chiu & Liu (2023) - JRC based on Oppikofer, JRC = 0.447 × a(mm) × L(m)^(-0.9515)

import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def calc_jrc_oppikofer(amplitude_m, length_m):
    """JRC from Oppikofer formula. Input: amplitude(m), length(m)."""
    if length_m <= 0 or amplitude_m <= 0:
        return 0
    amp_mm = amplitude_m * 1000
    jrc = 0.447 * amp_mm * (length_m ** -0.9515)
    return min(max(jrc, 0), 25)


def convert_to_jrc20(jrc, length_m):
    """Scale JRC to 20cm reference length (Barton et al. 1985)."""
    if length_m <= 0 or jrc <= 0:
        return 0
    L20 = 0.2
    if abs(length_m - L20) < 0.001:
        return jrc
    exp = -0.02 * jrc
    jrc20 = jrc * ((L20 / length_m) ** exp)
    return min(max(jrc20, 0), 25)


def jrc20_to_jr(jrc20):
    """Convert JRC20 to Jr via interpolation table (Table 2, Chiu & Liu 2023)."""
    table = [(0.5, 0.5), (1.5, 1.0), (2.5, 1.5), (7, 1.5), (11, 2), (14, 3), (20, 4)]
    if jrc20 <= 0.5:
        return 0.5
    if jrc20 >= 20:
        return 4.0
    for i in range(len(table) - 1):
        jrc1, jr1 = table[i]
        jrc2, jr2 = table[i + 1]
        if jrc1 <= jrc20 <= jrc2:
            jr = jr1 + (jr2 - jr1) * (jrc20 - jrc1) / (jrc2 - jrc1)
            return round(jr, 2)
    return 2.0


def get_description(jr):
    """ISRM roughness description from Jr."""
    if jr >= 3.5:
        return "Stepped rough"
    elif jr >= 2.5:
        return "Undulating rough / Stepped smooth"
    elif jr >= 1.75:
        return "Undulating smooth"
    elif jr >= 1.25:
        return "Planar rough / Undulating slickensided"
    elif jr >= 0.75:
        return "Planar smooth"
    return "Planar slickensided"


def dip_dipdir_to_normal(dip, dipdir):
    """Convert dip/dipdir to unit normal vector."""
    dr, dd = np.radians(dip), np.radians(dipdir)
    return np.array([np.sin(dr)*np.sin(dd), np.sin(dr)*np.cos(dd), np.cos(dr)])


def angle_between(dip1, dd1, dip2, dd2):
    """Angle between two orientations in degrees."""
    n1, n2 = dip_dipdir_to_normal(dip1, dd1), dip_dipdir_to_normal(dip2, dd2)
    dot = np.clip(np.dot(n1, n2), -1.0, 1.0)
    return np.degrees(np.arccos(abs(dot)))


def find_col(df, names):
    """Find column by possible names (case-insensitive)."""
    for name in names:
        for col in df.columns:
            if name.lower() == col.lower() or name.lower() in col.lower():
                return col
    return None


def main():
    print("\n" + "="*60)
    print("  Jr Calculator - Oppikofer Method, Optional for specific plane")
    print("  JRC = 0.447 × a(mm) × L(m)^(-0.9515)")
    print("="*60)
    
    root = tk.Tk()
    root.withdraw()
    
    # Load facets file
    print("\nSelect FACETS CSV file...")
    facets_path = filedialog.askopenfilename(
        title="Select FACETS CSV file",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    if not facets_path:
        print("No file selected.")
        root.destroy()
        return
    
    # Detect separator and load
    with open(facets_path) as f:
        sep = ';' if ';' in f.readline() else ','
    df = pd.read_csv(facets_path, sep=sep)
    df.columns = [c.strip() for c in df.columns]
    print(f"Loaded {len(df)} facets")
    
    # Find columns
    rms_col = find_col(df, ['RMS', 'rms'])
    area_col = find_col(df, ['Surface', 'Area'])
    dip_col = find_col(df, ['Dip'])
    dipdir_col = find_col(df, ['Dip dir.', 'Dip dir', 'DipDir'])
    
    if not rms_col or not area_col:
        print(f"Missing required columns! RMS={rms_col}, Area={area_col}")
        root.destroy()
        return
    print(f"Using: RMS='{rms_col}', Area='{area_col}'")
    
    # Optional spots file
    use_spots = messagebox.askyesno("RANSAC Spots", 
        "Do you have a RANSAC spots CSV to group facets by orientation?")
    
    spots_info = None
    angle_threshold = 15.0
    
    if use_spots:
        spots_path = filedialog.askopenfilename(
            title="Select RANSAC spots CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if spots_path:
            with open(spots_path) as f:
                sep_s = ';' if ';' in f.readline() else ','
            sdf = pd.read_csv(spots_path, sep=sep_s)
            sdf.columns = [c.strip() for c in sdf.columns]
            print(f"Loaded {len(sdf)} spots")
            
            dip_s = find_col(sdf, ['Dip'])
            dd_s = find_col(sdf, ['Dip dir', 'DipDir'])
            
            if dip_s and dd_s:
                spots_info = []
                print("\nSpot orientations:")
                for i, row in sdf.iterrows():
                    d, dd = float(row[dip_s]), float(row[dd_s])
                    spots_info.append({'index': i+1, 'dip': d, 'dipdir': dd})
                    print(f"  Spot {i+1}: {d:.0f}/{dd:.0f}")
                
                t = input("Angle threshold for matching (default 15): ").strip()
                angle_threshold = float(t) if t else 15.0
    
    root.destroy()
    
    # Process facets
    print(f"\nProcessing {len(df)} facets...")
    results = []
    
    for idx, row in df.iterrows():
        try:
            area = float(row[area_col])
            rms = float(row[rms_col])
            if area <= 0 or rms <= 0:
                continue
            
            length_m = np.sqrt(area)
            amp_m = 2 * rms
            
            jrc = calc_jrc_oppikofer(amp_m, length_m)
            jrc20 = convert_to_jrc20(jrc, length_m)
            jr = jrc20_to_jr(jrc20)
            
            r = {
                'Facet_Index': idx,
                'Area_m2': round(area, 6),
                'RMS_m': round(rms, 6),
                'Length_m': round(length_m, 4),
                'Amplitude_mm': round(amp_m * 1000, 3),
                'JRC': round(jrc, 2),
                'JRC20': round(jrc20, 2),
                'Jr': jr,
                'Description': get_description(jr)
            }
            
            if dip_col and dipdir_col:
                dip, dd = float(row[dip_col]), float(row[dipdir_col])
                r['Dip'] = round(dip, 1)
                r['Dip_Direction'] = round(dd, 1)
                
                if spots_info:
                    min_ang, matched = float('inf'), 0
                    for spot in spots_info:
                        ang = angle_between(dip, dd, spot['dip'], spot['dipdir'])
                        if ang < min_ang:
                            min_ang = ang
                            if ang <= angle_threshold:
                                matched = spot['index']
                    r['Spot'] = matched
                    r['Angle_Diff'] = round(min_ang, 2)
            
            results.append(r)
        except:
            continue
    
    print(f"Processed {len(results)} facets")
    if not results:
        print("No valid facets!")
        return
    
    rdf = pd.DataFrame(results)
    
    # Print statistics
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"\nJRC20: mean={rdf['JRC20'].mean():.2f}, median={rdf['JRC20'].median():.2f}")
    print(f"Jr: mean={rdf['Jr'].mean():.2f}, median={rdf['Jr'].median():.2f}")
    
    if spots_info and 'Spot' in rdf.columns:
        print("\nBy Spot:")
        print(f"{'Spot':<6} {'Orient':<12} {'N':<6} {'JRC20':<8} {'Jr':<6}")
        print("-"*45)
        spot_summaries = []
        for spot in spots_info:
            sd = rdf[rdf['Spot'] == spot['index']]
            n = len(sd)
            jrc20_med = round(sd['JRC20'].median(), 1) if n > 0 else 0
            jr_med = jrc20_to_jr(jrc20_med) if n > 0 else 0
            print(f"{spot['index']:<6} {spot['dip']:.0f}/{spot['dipdir']:.0f}    {n:<6} {jrc20_med:<8} {jr_med:<6}")
            spot_summaries.append({
                'Spot': spot['index'], 'Dip': spot['dip'], 'Dip_Direction': spot['dipdir'],
                'N_Facets': n, 'Median_JRC20': jrc20_med, 'Median_Jr': jr_med
            })
    
    rep_jrc20 = round(rdf['JRC20'].median(), 1)
    rep_jr = jrc20_to_jr(rep_jrc20)
    print(f"\nRepresentative: JRC20={rep_jrc20}, Jr={rep_jr} ({get_description(rep_jr)})")
    
    # Save results
    out_dir = os.path.dirname(facets_path)
    out_detail = os.path.join(out_dir, "Jr_specific_spot.csv")
    rdf.to_csv(out_detail, index=False)
    print(f"\nSaved: {out_detail}")
    
    if spots_info:
        out_sum = os.path.join(out_dir, "Jr_specific_spot.csv")
        pd.DataFrame(spot_summaries).to_csv(out_sum, index=False)
        print(f"Saved: {out_sum}")
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Jr Analysis - Oppikofer (n={len(rdf)}, Jr={rep_jr})')
    
    ax1.hist(rdf['JRC20'], bins=25, color='steelblue', edgecolor='white', alpha=0.7)
    ax1.axvline(rep_jrc20, color='red', ls='--', lw=2, label=f'Median={rep_jrc20}')
    ax1.set_xlabel('JRC20')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    ax2.hist(rdf['Jr'], bins=[0.5,1,1.5,2,2.5,3,3.5,4], color='coral', edgecolor='white', alpha=0.7)
    ax2.axvline(rep_jr, color='red', ls='--', lw=2, label=f'Jr={rep_jr}')
    ax2.set_xlabel('Jr')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(out_dir, "Jr_per_spot.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {plot_path}")
    plt.show()
    
    print("\nDone!")


if __name__ == "__main__":
    main()
