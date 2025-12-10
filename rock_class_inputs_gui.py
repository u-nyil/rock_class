# rock_class_calc_gui.py
# Python 3.10.11, Windows 11 — Tkinter GUI
# Enter parameters for Q-System, RMR, and RMi. Computes index and category.

from __future__ import annotations

import math
import tkinter as tk
from tkinter import ttk, messagebox


# -----------------------------
# Q-System choice dictionaries
# -----------------------------

Q_JN_CHOICES = [
    ("0.5  Massive (lower bound)", 0.5),
    ("1.0  Massive (upper bound)", 1.0),
    ("2    One joint set", 2.0),
    ("3    One set + random", 3.0),
    ("4    Two sets", 4.0),
    ("6    Two sets + random", 6.0),
    ("9    Three sets", 9.0),
    ("12   Three sets + random", 12.0),
    ("15   ≥4 sets / heavily jointed", 15.0),
    ("20   Crushed rock / earthlike", 20.0),
]

Q_JR_CHOICES = [
    ("4.0   Discontinuous joints (very rough)", 4.0),
    ("3.0   Rough/irregular undulating", 3.0),
    ("2.0   Smooth undulating", 2.0),
    ("1.5   Slickensided undulating", 1.5),
    ("1.5   Rough/irregular planar", 1.5),
    ("1.0   Smooth planar", 1.0),
    ("0.5   Slickensided planar", 0.5),
    ("1.0   No wall contact when sheared (nominal)", 1.0),
]

Q_JA_CHOICES = [
    ("0.75  Healed/welded", 0.75),
    ("1.0   Unaltered walls, staining only", 1.0),
    ("2.0   Slightly altered, clay-free coatings", 2.0),
    ("3.0   Silty/sandy-clay (non-softening)", 3.0),
    ("4.0   Soft/low-friction clay (discontinuous)", 4.0),
    ("4.0   Sandy particles, clay-free, disintegrating", 4.0),
    ("6.0   Strongly OC non-softening clay <5 mm", 6.0),
    ("8.0   Medium/low OC softening clay <5 mm", 8.0),
    ("8.0   Swelling clay ≤5 mm (lower)", 8.0),
    ("12.0  Swelling clay ≤5 mm (upper)", 12.0),
    # No wall contact variants:
    ("6.0   Disintegrated/crushed rock zones (no wall contact)", 6.0),
    ("8.0   Rock + clay (lower, no wall contact)", 8.0),
    ("12.0  Rock + clay (upper, no wall contact)", 12.0),
    ("5.0   Thick continuous clay zones (no wall contact)", 5.0),
    ("24.0  Clay conditions (upper, no wall contact)", 24.0),
]

Q_JW_CHOICES = [
    ("1.00  Dry / minor inflow (<5 L/min locally)", 1.0),
    ("0.66  Medium inflow or occasional outwash", 0.66),
    ("0.50  Large inflow/high pressure (competent rock)", 0.5),
    ("0.33  Large inflow/high pressure", 0.33),
    ("0.20  Exceptionally high inflow/pressure (decaying)", 0.2),
    ("0.10  Exceptionally high inflow/pressure (variant)", 0.1),
    ("0.05  Extreme inflow/pressure", 0.05),
]

Q_SRF_CHOICES = [
    ("0.5   High stress, very tight structure (lower)", 0.5),
    ("1.0   Medium stress", 1.0),
    ("2.0   High stress, very tight (upper)", 2.0),
    ("2.5   Low stress near surface", 2.5),
    ("5.0   Mild rockburst OR loose/open joints", 5.0),
    ("7.5   Multiple clay-free shear zones; loose rock", 7.5),
    ("10.0  Multiple weakness zones with clay/disintegration", 10.0),
    ("15.0  Squeezing (upper mid)", 15.0),
    ("20.0  Heavy rockburst / heavy squeezing", 20.0),
]


# --------------------------------
# RMR (1989) choice dictionaries
# --------------------------------

RMR_UCS_CHOICES = [
    ("15  UCS > 250 MPa", 15),
    ("12  100–250 MPa", 12),
    ("7   50–100 MPa", 7),
    ("4   25–50 MPa", 4),
    ("2   5–25 MPa", 2),
    ("1   1–5 MPa", 1),
    ("0   <1 MPa", 0),
]

RMR_RQD_CHOICES = [
    ("20  RQD 90–100 %", 20),
    ("17  75–90 %", 17),
    ("13  50–75 %", 13),
    ("8   25–50 %", 8),
    ("3   <25 %", 3),
]

RMR_SPACING_CHOICES = [
    ("20  >2 m", 20),
    ("15  0.6–2 m", 15),
    ("10  0.2–0.6 m", 10),
    ("8   0.06–0.2 m", 8),
    ("5   <0.06 m", 5),
]

RMR_PERSISTENCE_CHOICES = [
    ("6  <1 m", 6),
    ("4  1–3 m", 4),
    ("2  3–10 m", 2),
    ("1  10–20 m", 1),
    ("0  >20 m", 0),
]

RMR_APERTURE_CHOICES = [
    ("6  0 (tight)", 6),
    ("5  <0.1 mm", 5),
    ("4  0.1–1.0 mm", 4),
    ("1  1–5 mm", 1),
    ("0  >5 mm", 0),
]

RMR_ROUGHNESS_CHOICES = [
    ("6  Very rough", 6),
    ("5  Rough", 5),
    ("3  Slightly rough", 3),
    ("1  Smooth", 1),
    ("0  Slickensided", 0),
]

RMR_INFILL_CHOICES = [
    ("6  None", 6),
    ("5  Hard filling <5 mm", 5),
    ("2  Hard filling >5 mm", 2),
    ("1  Soft filling ≤5 mm", 1),
    ("0  Soft filling >5 mm", 0),
]

RMR_WEATHERING_CHOICES = [
    ("6  Unweathered", 6),
    ("5  Slightly weathered", 5),
    ("3  Moderately weathered", 3),
    ("1  Highly weathered", 1),
    ("0  Decomposed", 0),
]

RMR_GW_CHOICES = [
    ("15  None / dry / σw/σ1=0", 15),
    ("10  <10 L/min/10 m  OR damp  OR <0.1", 10),
    ("7   10–25 L/min/10 m OR wet   OR 0.1–0.2", 7),
    ("4   25–125 L/min/10 m OR dripping OR 0.2–0.5", 4),
    ("0   >125 L/min/10 m  OR flowing  OR >0.5", 0),
]

RMR_ORI_CHOICES_TUNNEL = [
    ("0    Very favourable", 0),
    ("-2   Favourable", -2),
    ("-5   Fair", -5),
    ("-10  Unfavourable", -10),
    ("-12  Very unfavourable", -12),
]


# --------------------------------
# Helper functions (calculations)
# --------------------------------

def safe_float(s: str, default: float | None = None) -> float | None:
    try:
        return float(s)
    except Exception:
        return default

def q_value(rqd: float, jn: float, jr: float, ja: float, jw: float, srf: float) -> float:
    # Classic Barton form (RQD is used as %, not fraction)
    return (rqd / max(jn, 1e-9)) * (jr / max(ja, 1e-9)) * (jw / max(srf, 1e-9))

def q_category(q: float) -> str:
    if q < 0.001:
        return "Exceptionally poor"
    if q < 0.01:
        return "Exceptionally poor"
    if q < 0.1:
        return "Extremely poor"
    if q < 1:
        return "Very poor"
    if q < 4:
        return "Poor"
    if q < 10:
        return "Fair"
    if q < 40:
        return "Good"
    return "Very good"

def rmr_value(ucs_rating: float, rqd_rating: float, spacing_rating: float,
              pers: float, aper: float, rgh: float, infi: float, weath: float,
              gw_rating: float, orientation_adj: float) -> float:
    joint_cond = pers + aper + rgh + infi + weath
    base = ucs_rating + rqd_rating + spacing_rating + joint_cond + gw_rating
    return base + orientation_adj

def rmr_category(rmr: float) -> str:
    if rmr >= 81:
        return "Class I — Very Good"
    if rmr >= 61:
        return "Class II — Good"
    if rmr >= 41:
        return "Class III — Fair"
    if rmr >= 21:
        return "Class IV — Poor"
    return "Class V — Very Poor"

# -----------------------------
# Enhanced RMi helpers
# -----------------------------

JR_CATEGORY_REFERENCE: list[tuple[float, str]] = [
    (4.0, "Discontinuous joints (very rough)"),
    (3.0, "Rough/irregular undulating"),
    (2.0, "Smooth undulating"),
    (1.5, "Slickensided/rough planar"),
    (1.0, "Smooth planar / nominal wall contact"),
    (0.5, "Slickensided planar / very smooth"),
]

def jr_category_from_value(jr: float) -> tuple[str, float | None]:
    """
    Map a numeric Jr to the closest standard descriptor.
    Returns (label, reference_value). For non-positive Jr, label explains the issue.
    """
    if jr is None or jr <= 0:
        return ("Enter Jr > 0 to categorize", None)
    ref_val, label = min(JR_CATEGORY_REFERENCE, key=lambda pair: abs(pair[0] - jr))
    return (label, ref_val)

def rmi_components(sigma_c_mpa: float, vb_m3: float, jR: float, jA: float, jL: float) -> dict[str, float | bool]:
    """
    Detailed RMi calculation (Palmstrom):
      jC = jR * jL / jA
      D  = 0.37 * jC - 0.2
      JP = 0.2 * jC * Vb^D
      RMi = sigma_c * JP
    """
    valid = sigma_c_mpa > 0 and vb_m3 > 0 and jR > 0 and jA > 0 and jL > 0
    if not valid:
        return {
            "valid": False,
            "jc": 0.0,
            "d_exp": 0.0,
            "jp": 0.0,
            "rmi": 0.0,
        }

    jc = (jR * jL) / jA
    d_exp = (0.37 * jc) - 0.2
    jp = 0.2 * jc * (vb_m3 ** d_exp)
    rmi = sigma_c_mpa * jp

    return {
        "valid": True,
        "jc": jc,
        "d_exp": d_exp,
        "jp": jp,
        "rmi": rmi,
    }

def rmi_value(sigma_c_mpa: float, vb_m3: float, jR: float, jA: float, jL: float) -> float:
    """Return only the RMi value (MPa); wraps rmi_components."""
    return float(rmi_components(sigma_c_mpa, vb_m3, jR, jA, jL).get("rmi", 0.0))


def nearest_choice(value: float, choices: list[tuple[str, float]]) -> tuple[float, str]:
    """
    Given a numeric value and a list of (label, numeric) choices, return (numeric, label)
    for the closest numeric entry. If choices is empty, returns (value, "").
    """
    if not choices:
        return value, ""
    best_label, best_val = min(((lbl, val) for lbl, val in choices), key=lambda x: abs(x[1] - value))
    return best_val, best_label

def rmr_category(rmr: float) -> str:
    if rmr >= 81:
        return "Class I - Very Good"
    if rmr >= 61:
        return "Class II - Good"
    if rmr >= 41:
        return "Class III - Fair"
    if rmr >= 21:
        return "Class IV - Poor"
    return "Class V - Very Poor"


# -----------------------------
# GUI
# -----------------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Rock Mass Classification Calculator — Q / RMR / RMi")
        self.geometry("1100x780")
        self.minsize(1000, 720)

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=8, pady=8)

        self.q_tab = ttk.Frame(nb)
        self.rmr_tab = ttk.Frame(nb)
        self.rmi_tab = ttk.Frame(nb)
        nb.add(self.q_tab, text="Q-System (Barton)")
        nb.add(self.rmr_tab, text="RMR (Bieniawski)")
        nb.add(self.rmi_tab, text="RMi (Palmström)")

        self.build_q_tab(self.q_tab)
        self.build_rmr_tab(self.rmr_tab)
        self.build_rmi_tab(self.rmi_tab)

    # -------- Q TAB --------
    def build_q_tab(self, parent: ttk.Frame):
        frm = ttk.Frame(parent)
        frm.pack(fill="both", expand=True, padx=6, pady=6)

        grid = ttk.Frame(frm)
        grid.pack(side="top", anchor="nw", pady=(0, 8))

        r = 0
        ttk.Label(grid, text="RQD (%)").grid(row=r, column=0, sticky="e", padx=4, pady=4)
        self.q_rqd = ttk.Entry(grid, width=12)
        self.q_rqd.insert(0, "85")
        self.q_rqd.grid(row=r, column=1, sticky="w", padx=4, pady=4)

        r += 1
        ttk.Label(grid, text="Jn (choose)").grid(row=r, column=0, sticky="e", padx=4, pady=4)
        self.q_jn = ttk.Combobox(grid, width=40, values=[t for t, _ in Q_JN_CHOICES], state="readonly")
        self.q_jn.current(5)  # default 6
        self.q_jn.grid(row=r, column=1, columnspan=2, sticky="w", padx=4, pady=4)

        r += 1
        ttk.Label(grid, text="Jr (numeric)").grid(row=r, column=0, sticky="e", padx=4, pady=4)
        self.q_jr_entry = ttk.Entry(grid, width=12)
        self.q_jr_entry.insert(0, "3")
        self.q_jr_entry.grid(row=r, column=1, sticky="w", padx=4, pady=4)

        r += 1
        ttk.Label(grid, text="Ja (choose)").grid(row=r, column=0, sticky="e", padx=4, pady=4)
        self.q_ja = ttk.Combobox(grid, width=60, values=[t for t, _ in Q_JA_CHOICES], state="readonly")
        self.q_ja.current(1)  # 1.0 unaltered
        self.q_ja.grid(row=r, column=1, columnspan=3, sticky="w", padx=4, pady=4)

        r += 1
        ttk.Label(grid, text="Jw (choose)").grid(row=r, column=0, sticky="e", padx=4, pady=4)
        self.q_jw = ttk.Combobox(grid, width=60, values=[t for t, _ in Q_JW_CHOICES], state="readonly")
        self.q_jw.current(0)
        self.q_jw.grid(row=r, column=1, columnspan=3, sticky="w", padx=4, pady=4)

        r += 1
        ttk.Label(grid, text="SRF (choose)").grid(row=r, column=0, sticky="e", padx=4, pady=4)
        self.q_srf = ttk.Combobox(grid, width=60, values=[t for t, _ in Q_SRF_CHOICES], state="readonly")
        self.q_srf.current(3)  # 2.5 low stress near surface
        self.q_srf.grid(row=r, column=1, columnspan=3, sticky="w", padx=4, pady=4)

        ttk.Button(frm, text="Compute Q", command=self.compute_q).pack(anchor="w", padx=6, pady=(6, 4))

        self.q_out = tk.Text(frm, height=7, wrap="word", padx=8, pady=8)
        self.q_out.pack(fill="x", expand=False)
        self.q_out.configure(state="disabled")

    def compute_q(self):
        rqd = safe_float(self.q_rqd.get(), 0.0) or 0.0
        jn = Q_JN_CHOICES[self.q_jn.current()][1]
        jr_input = safe_float(self.q_jr_entry.get(), 0.0) or 0.0
        jr_mapped, jr_label = nearest_choice(jr_input, Q_JR_CHOICES)
        jr_used = jr_input  # use the raw numeric Jr entered by the user
        ja = Q_JA_CHOICES[self.q_ja.current()][1]
        jw = Q_JW_CHOICES[self.q_jw.current()][1]
        srf = Q_SRF_CHOICES[self.q_srf.current()][1]

        q = q_value(rqd, jn, jr_used, ja, jw, srf)
        cat = q_category(q)

        self.q_out.configure(state="normal")
        self.q_out.delete("1.0", "end")
        self.q_out.insert(
            "1.0",
            "Inputs:\n"
            f"  RQD={rqd:.2f} %, Jn={jn:g}, Jr(raw)={jr_used:.3g} (nearest catalog {jr_mapped:g}: {jr_label}), "
            f"Ja={ja:g}, Jw={jw:g}, SRF={srf:g}\n\n"
            f"Result:\n  Q = {q:.5g}\n  Category: {cat}\n"
        )
        self.q_out.configure(state="disabled")

    # -------- RMR TAB --------
    def build_rmr_tab(self, parent: ttk.Frame):
        frm = ttk.Frame(parent)
        frm.pack(fill="both", expand=True, padx=6, pady=6)

        grid = ttk.Frame(frm)
        grid.pack(side="top", anchor="nw", pady=(0, 8))

        # Build all dropdowns with labels
        row = 0
        def add_combo(label, choices, default_idx=0):
            nonlocal row
            ttk.Label(grid, text=label).grid(row=row, column=0, sticky="e", padx=4, pady=4)
            cb = ttk.Combobox(grid, width=40, values=[t for t, _ in choices], state="readonly")
            cb.current(default_idx)
            cb.grid(row=row, column=1, sticky="w", padx=4, pady=4)
            row += 1
            return cb

        self.rmr_ucs = add_combo("Intact rock strength (rating)", RMR_UCS_CHOICES, 1)
        self.rmr_rqd = add_combo("RQD (rating)", RMR_RQD_CHOICES, 2)
        self.rmr_spacing = add_combo("Discontinuity spacing (rating)", RMR_SPACING_CHOICES, 1)

        ttk.Label(grid, text="Joint condition subfactors").grid(row=row, column=0, columnspan=2, sticky="w", padx=4, pady=(10, 0))
        row += 1
        self.rmr_pers = add_combo("  Persistence", RMR_PERSISTENCE_CHOICES, 2)
        self.rmr_apert = add_combo("  Aperture", RMR_APERTURE_CHOICES, 2)
        ttk.Label(grid, text="  Roughness (enter Jr, auto category)").grid(row=row, column=0, sticky="e", padx=4, pady=4)
        self.rmr_rough_entry = ttk.Entry(grid, width=12)
        self.rmr_rough_entry.insert(0, "3")
        self.rmr_rough_entry.grid(row=row, column=1, sticky="w", padx=4, pady=4)
        row += 1
        self.rmr_infill = add_combo("  Infilling", RMR_INFILL_CHOICES, 1)
        self.rmr_weath = add_combo("  Weathering", RMR_WEATHERING_CHOICES, 1)

        self.rmr_gw = add_combo("Groundwater (rating)", RMR_GW_CHOICES, 3)
        self.rmr_ori = add_combo("Orientation adjustment (tunnels & mines)", RMR_ORI_CHOICES_TUNNEL, 2)

        ttk.Button(frm, text="Compute RMR", command=self.compute_rmr).pack(anchor="w", padx=6, pady=(6, 4))

        self.rmr_out = tk.Text(frm, height=8, wrap="word", padx=8, pady=8)
        self.rmr_out.pack(fill="x", expand=False)
        self.rmr_out.configure(state="disabled")

    def compute_rmr(self):
        rough_input = safe_float(self.rmr_rough_entry.get(), 0.0) or 0.0
        vals = {
            "UCS": RMR_UCS_CHOICES[self.rmr_ucs.current()][1],
            "RQD": RMR_RQD_CHOICES[self.rmr_rqd.current()][1],
            "Spacing": RMR_SPACING_CHOICES[self.rmr_spacing.current()][1],
            "Pers": RMR_PERSISTENCE_CHOICES[self.rmr_pers.current()][1],
            "Aper": RMR_APERTURE_CHOICES[self.rmr_apert.current()][1],
            "Rough": rough_input,
            "Inf": RMR_INFILL_CHOICES[self.rmr_infill.current()][1],
            "Weath": RMR_WEATHERING_CHOICES[self.rmr_weath.current()][1],
            "GW": RMR_GW_CHOICES[self.rmr_gw.current()][1],
            "Ori": RMR_ORI_CHOICES_TUNNEL[self.rmr_ori.current()][1],
        }
        rough_mapped, rough_label = nearest_choice(vals["Rough"], RMR_ROUGHNESS_CHOICES)
        vals["Rough"] = rough_mapped
        rmr = rmr_value(
            ucs_rating=vals["UCS"],
            rqd_rating=vals["RQD"],
            spacing_rating=vals["Spacing"],
            pers=vals["Pers"],
            aper=vals["Aper"],
            rgh=vals["Rough"],
            infi=vals["Inf"],
            weath=vals["Weath"],
            gw_rating=vals["GW"],
            orientation_adj=vals["Ori"],
        )
        cat = rmr_category(rmr)
        gsi_est = rmr - 5  # common heuristic: GSI ≈ RMR - 5

        self.rmr_out.configure(state="normal")
        self.rmr_out.delete("1.0", "end")
        self.rmr_out.insert(
            "1.0",
            "Selected ratings:\n"
            f"  UCS={vals['UCS']}, RQD={vals['RQD']}, Spacing={vals['Spacing']}, "
            f"Pers={vals['Pers']}, Aper={vals['Aper']}, Rough={vals['Rough']} (from Jr_in={rough_input:.3g} -> {rough_label}), "
            f"Inf={vals['Inf']}, Weath={vals['Weath']}, GW={vals['GW']}, Ori={vals['Ori']}\n\n"
            f"Result:\n  RMR = {rmr:.1f}\n  Category: {cat}\n"
            f"  GSI (approx, RMR-5) = {gsi_est:.1f}\n"
        )
        self.rmr_out.configure(state="disabled")

    # -------- RMi TAB --------
    def build_rmi_tab(self, parent: ttk.Frame):
        frm = ttk.Frame(parent)
        frm.pack(fill="both", expand=True, padx=6, pady=6)

        grid = ttk.Frame(frm)
        grid.pack(side="top", anchor="nw", pady=(0, 8), fill="x")

        ttk.Label(grid, text="UCS (sigma_c, MPa)").grid(row=0, column=0, sticky="e", padx=4, pady=4)
        self.rmi_sigma_entry = ttk.Entry(grid, width=12)
        self.rmi_sigma_entry.insert(0, "150")
        self.rmi_sigma_entry.grid(row=0, column=1, sticky="w", padx=4, pady=4)

        ttk.Label(grid, text="Jr (numeric roughness)").grid(row=1, column=0, sticky="e", padx=4, pady=4)
        self.rmi_jr_entry = ttk.Entry(grid, width=12)
        self.rmi_jr_entry.insert(0, "3")
        self.rmi_jr_entry.grid(row=1, column=1, sticky="w", padx=4, pady=4)

        ttk.Label(grid, text="jA (alteration)").grid(row=2, column=0, sticky="e", padx=4, pady=4)
        self.rmi_ja_entry = ttk.Entry(grid, width=12)
        self.rmi_ja_entry.insert(0, "4")
        self.rmi_ja_entry.grid(row=2, column=1, sticky="w", padx=4, pady=4)

        ttk.Label(grid, text="jL (size/length)").grid(row=3, column=0, sticky="e", padx=4, pady=4)
        self.rmi_jl_entry = ttk.Entry(grid, width=12)
        self.rmi_jl_entry.insert(0, "2")
        self.rmi_jl_entry.grid(row=3, column=1, sticky="w", padx=4, pady=4)

        ttk.Label(grid, text="Spacings (m) -> Vb product").grid(row=4, column=0, sticky="ne", padx=4, pady=4)
        spacing_controls = ttk.Frame(grid)
        spacing_controls.grid(row=4, column=1, sticky="w", padx=4, pady=4)

        self.rmi_spacing_entries: list[ttk.Entry] = []
        self.rmi_spacing_frame = ttk.Frame(spacing_controls)
        self.rmi_spacing_frame.pack(anchor="w")

        def add_spacing_entry(default_val: str = "0.5"):
            idx = len(self.rmi_spacing_entries) + 1
            row = ttk.Frame(self.rmi_spacing_frame)
            row.pack(anchor="w", pady=2)
            ttk.Label(row, text=f"Spacing {idx}:").pack(side="left")
            ent = ttk.Entry(row, width=10)
            ent.insert(0, default_val)
            ent.pack(side="left", padx=(4, 0))
            self.rmi_spacing_entries.append(ent)

        ttk.Button(spacing_controls, text="Add spacing", command=lambda: add_spacing_entry("0.5")).pack(anchor="w", pady=(2, 0))
        add_spacing_entry("0.3")

        ttk.Label(
            frm,
            text="Tip: Jr uses the raw numeric value (nearest category shown for reference). Vb is the product of positive spacings.",
        ).pack(anchor="w", padx=6, pady=(4, 0))
        ttk.Button(frm, text="Compute RMi", command=self.compute_rmi).pack(anchor="w", padx=6, pady=(6, 4))

        self.rmi_out = tk.Text(frm, height=16, wrap="word", padx=8, pady=8)
        self.rmi_out.pack(fill="x", expand=False)
        self.rmi_out.configure(state="disabled")

    def compute_rmi(self):
        sigma_c = safe_float(self.rmi_sigma_entry.get(), 0.0) or 0.0
        jR = safe_float(self.rmi_jr_entry.get(), 0.0) or 0.0
        jA = safe_float(self.rmi_ja_entry.get(), 1.0) or 1.0
        jL = safe_float(self.rmi_jl_entry.get(), 1.0) or 1.0

        spacing_vals = []
        for ent in getattr(self, "rmi_spacing_entries", []):
            val = safe_float(ent.get(), None)
            if val is not None and val > 0:
                spacing_vals.append(val)

        vb = 0.0
        if spacing_vals:
            vb = 1.0
            for v in spacing_vals:
                vb *= v

        jr_label, jr_ref = jr_category_from_value(jR)
        comp = rmi_components(sigma_c, vb, jR, jA, jL)

        self.rmi_out.configure(state="normal")
        self.rmi_out.delete("1.0", "end")

        if not comp.get("valid"):
            self.rmi_out.insert(
                "1.0",
                "Enter positive values for sigma_c, Jr, jA, jL, and at least one positive spacing to compute RMi.",
            )
            self.rmi_out.configure(state="disabled")
            return

        spacing_txt = ", ".join(f"{v:.3g}" for v in spacing_vals) if spacing_vals else "None"

        lines = [
            "Inputs:",
            f"  sigma_c = {sigma_c:.3g} MPa",
            f"  Spacings (m) = [{spacing_txt}]",
            f"  Vb = product(spacings) = {vb:.3g} m^3",
        ]

        jr_line = f"  Jr(raw) = {jR:.3g}"
        if jr_label:
            if jr_ref is not None:
                jr_line += f"  (approx: {jr_label}; nearest standard {jr_ref:g})"
            else:
                jr_line += f"  ({jr_label})"
        lines.append(jr_line)
        lines.append(f"  jL = {jL:.3g}, jA = {jA:.3g}")

        jc = comp["jc"]
        d_exp = comp["d_exp"]
        jp = comp["jp"]
        rmi = comp["rmi"]

        lines.extend(
            [
                "",
                "Steps:",
                f"  jC = Jr * jL / jA = {jR:.3g} * {jL:.3g} / {jA:.3g} = {jc:.3g}",
                f"  D  = 0.37 * jC - 0.2 = 0.37 * {jc:.3g} - 0.2 = {d_exp:.3g}",
                f"  JP = 0.2 * jC * Vb^D = 0.2 * {jc:.3g} * {vb:.3g}^{d_exp:.3g} = {jp:.3g}",
                "",
                "Result:",
                f"  RMi = sigma_c * JP = {sigma_c:.3g} * {jp:.3g} = {rmi:.3g} MPa",
            ]
        )

        self.rmi_out.insert("1.0", "\n".join(lines))
        self.rmi_out.configure(state="disabled")


# -----------
# Main
# -----------

def main():
    try:
        App().mainloop()
    except Exception as e:
        tk.Tk().withdraw()
        messagebox.showerror("Error", f"An unexpected error occurred:\n{e}")

if __name__ == "__main__":
    main()
