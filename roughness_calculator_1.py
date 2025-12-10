# Jr Calculator
# Uses histogram mode (0.1 interval) as representative JRC20
# Reference: Chiu, C.C., Liu, C., 2023. Development of a computer program from photogrammetry for assisting Q-system rating

import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def calc_jrc_oppikofer(amp_m, length_m):
    """JRC = 0.447 * a(mm) * L(m)^(-0.9515)"""
    if length_m <= 0 or amp_m <= 0:
        return 0
    amp_mm = amp_m * 1000
    jrc = 0.447 * amp_mm * (length_m ** -0.9515)
    return min(max(jrc, 0), 25)


def convert_to_jrc20(jrc, length_m):
    """Scale JRC to 20cm reference (Barton et al. 1985)."""
    if length_m <= 0 or jrc <= 0:
        return 0
    L20 = 0.2
    jrc20 = jrc * ((L20 / length_m) ** (-0.02 * jrc))
    return min(max(jrc20, 0), 25)


def jrc20_to_jr(jrc20):
    """Convert JRC20 to Jr via interpolation (Table 2)."""
    table = [(0.5, 0.5), (1.5, 1.0), (2.5, 1.5), (7, 1.5), (11, 2), (14, 3), (20, 4)]
    if jrc20 <= 0.5:
        return 0.5
    if jrc20 >= 20:
        return 4.0
    for i in range(len(table) - 1):
        jrc1, jr1 = table[i]
        jrc2, jr2 = table[i + 1]
        if jrc1 <= jrc20 <= jrc2:
            return round(jr1 + (jr2 - jr1) * (jrc20 - jrc1) / (jrc2 - jrc1), 2)
    return 2.0


def get_description(jr):
    """ISRM roughness description."""
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


def find_col(df, names):
    """Find column by possible names."""
    for name in names:
        for col in df.columns:
            if name.lower() in col.lower():
                return col
    return None


def main():
    print("\n" + "="*60)
    print("  Jr Calculator - Chiu & Liu (2023) Exact Method")
    print("  Uses histogram MODE (0.1 interval) as representative")
    print("="*60)
    
    root = tk.Tk()
    root.withdraw()
    
    print("\nSelect FACETS CSV file...")
    facets_path = filedialog.askopenfilename(
        title="Select FACETS CSV file",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    root.destroy()
    
    if not facets_path:
        print("No file selected.")
        return
    
    # Load file
    with open(facets_path) as f:
        sep = ';' if ';' in f.readline() else ','
    df = pd.read_csv(facets_path, sep=sep)
    df.columns = [c.strip() for c in df.columns]
    print(f"Loaded {len(df)} facets")
    
    # Find columns
    rms_col = find_col(df, ['RMS', 'rms'])
    area_col = find_col(df, ['Surface', 'Area'])
    
    if not rms_col or not area_col:
        print(f"Missing columns! RMS={rms_col}, Area={area_col}")
        return
    print(f"Using: RMS='{rms_col}', Area='{area_col}'")
    
    # Calculate JRC20 for all facets
    print(f"\nCalculating JRC20 for {len(df)} facets...")
    jrc20_list = []
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
            
            jrc20_list.append(jrc20)
            results.append({
                'Facet_Index': idx,
                'Area_m2': round(area, 6),
                'RMS_m': round(rms, 6),
                'Length_m': round(length_m, 4),
                'Amplitude_mm': round(amp_m * 1000, 3),
                'JRC': round(jrc, 2),
                'JRC20': round(jrc20, 2),
                'Jr': jrc20_to_jr(jrc20)
            })
        except:
            continue
    
    jrc20_arr = np.array(jrc20_list)
    print(f"Processed {len(jrc20_arr)} facets")
    
    # Find mode using histogram (paper method)
    print("\nFinding histogram mode (0.1 interval)...")
    bin_width = 0.1
    bins = np.arange(0, 25 + bin_width, bin_width)
    hist, edges = np.histogram(jrc20_arr, bins=bins)
    
    max_idx = np.argmax(hist)
    mode_jrc20 = (edges[max_idx] + edges[max_idx + 1]) / 2
    mode_jr = jrc20_to_jr(mode_jrc20)
    
    print(f"  Mode bin: {edges[max_idx]:.1f}-{edges[max_idx+1]:.1f}, count={hist[max_idx]}")
    
    # Statistics
    mean_jrc20 = jrc20_arr.mean()
    median_jrc20 = np.median(jrc20_arr)
    
    print("\n" + "="*60)
    print("STATISTICS")
    print("="*60)
    print(f"\nJRC20: mean={mean_jrc20:.2f}, median={median_jrc20:.2f}, mode={mode_jrc20:.1f}")
    print(f"       std={jrc20_arr.std():.2f}, range={jrc20_arr.min():.2f}-{jrc20_arr.max():.2f}")
    
    mean_jr = jrc20_to_jr(mean_jrc20)
    median_jr = jrc20_to_jr(median_jrc20)
    print(f"\nJr: from_mean={mean_jr:.2f}, from_median={median_jr:.2f}, from_mode={mode_jr:.2f}")
    
    # Category distribution
    print("\nDistribution by category:")
    cats = [(0, 2.5, "Planar"), (2.5, 7, "Undulating slickensided"),
            (7, 11, "Undulating smooth"), (11, 14, "Undulating rough"),
            (14, 20, "Stepped"), (20, 25, "Very rough")]
    for lo, hi, name in cats:
        cnt = np.sum((jrc20_arr >= lo) & (jrc20_arr < hi))
        pct = 100 * cnt / len(jrc20_arr)
        print(f"  {name:<25} {lo:.0f}-{hi:.0f}: {cnt:>5} ({pct:.1f}%)")
    
    # Representative values
    print("\n" + "="*60)
    print(f"REPRESENTATIVE (Paper Method)")
    print(f"  JRC20 = {mode_jrc20:.1f}")
    print(f"  Jr = {mode_jr:.2f} ({get_description(mode_jr)})")
    print("="*60)
    
    # Save results
    out_dir = os.path.dirname(facets_path)
    
    rdf = pd.DataFrame(results)
    out_detail = os.path.join(out_dir, "Jr_1_detailed.csv")
    rdf.to_csv(out_detail, index=False)
    print(f"\nSaved: {out_detail}")
    
    summary = {
        'Method': 'Chiu & Liu (2023) - Mode',
        'Total_Facets': len(jrc20_arr),
        'JRC20_Mean': round(mean_jrc20, 2),
        'JRC20_Median': round(median_jrc20, 2),
        'JRC20_Mode': round(mode_jrc20, 1),
        'Jr_from_Mode': mode_jr,
        'Description': get_description(mode_jr)
    }
    out_sum = os.path.join(out_dir, "Jr_1_summary.csv")
    pd.DataFrame([summary]).to_csv(out_sum, index=False)
    print(f"Saved: {out_sum}")
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1.hist(jrc20_arr, bins=50, color='steelblue', edgecolor='white', alpha=0.8, density=True)
    ax1.axvline(mode_jrc20, color='red', ls='--', lw=2)
    ax1.annotate(f'JRC20 = {mode_jrc20:.1f}', xy=(mode_jrc20, ax1.get_ylim()[1]*0.8),
                 xytext=(mode_jrc20+3, ax1.get_ylim()[1]*0.85), fontsize=11, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
                 arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax1.set_xlabel('JRC20')
    ax1.set_ylabel('Normalized ratio')
    ax1.set_title('JRC20 Distribution')
    ax1.set_xlim(0, 25)
    ax1.grid(alpha=0.3)
    
    # Comparison bar
    methods = ['Mean', 'Median', 'Mode\n(Paper)']
    jr_vals = [mean_jr, median_jr, mode_jr]
    bars = ax2.bar(methods, jr_vals, color=['#ff9999', '#99ff99', '#6699ff'], 
                   edgecolor='black', width=0.5)
    bars[2].set_edgecolor('red')
    bars[2].set_linewidth(3)
    for bar, val in zip(bars, jr_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f'{val:.2f}', ha='center', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Jr')
    ax2.set_title('Jr by Method')
    ax2.set_ylim(0, 4.5)
    ax2.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    out_plot = os.path.join(out_dir, "Jr_1.png")
    plt.savefig(out_plot, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_plot}")
    plt.show()
    
    print("\nDone!")


if __name__ == "__main__":
    main()
