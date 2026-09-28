"""Menu 3 — Accès aux services financiers : couverture territoriale des établissements formels."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from components import charts as ch
from components import territorial_map as tmap
from components.kpi import kpi_row
from components.layout import card_title, context_bar, empty_state, page_header, plot, source_note
from utils import config as C
from utils import metrics as M
from utils.data_loader import Datasets
from utils.filters import Filters
from utils.filters import density_chip as density_chip_labels
from utils.formatting import fmt_dec, fmt_int

_CLASSE_COLORS = {"Correct": C.COLORS["green"], "Faible": C.OPERATEUR_COLORS["Moov"], "MM-only": C.COLORS["red"]}


def render(ds: Datasets, f: Filters) -> None:
    page_header("Accès aux services financiers", "Couverture territoriale des banques, IMF, assurances et mutuelles.")
    basis = C.DENSITY_BASES[f.density_basis]
    formel = f.density_basis == "formel"
    density_chip = density_chip_labels(f)
    context_bar([
        ("Région", list(f.region)), ("Préfecture", list(f.prefecture)), ("Commune", list(f.commune)),
        ("Type d'établissement", list(f.etab_categorie)), ("Indicateur", [basis["label"]]),
        ("Densité", density_chip),
    ])

    kpis = M.financial_kpis(ds, f)
    if kpis:
        kpi_row(kpis)

    map_col, visual_col = st.columns(2, gap="medium")
    with map_col:
        with st.container(key="card_sf_map"):
            note_bits = []
            if f.etab_categorie:
                note_bits.append("Type : " + ", ".join(f.etab_categorie))
            if f.density_active:
                note_bits.append("Densité : niveau" + ("x " if len(f.density_levels) > 1 else " ")
                                 + ", ".join(map(str, f.density_levels)))
            note = " · ".join(note_bits) if note_bits else \
                ("Banques, IMF, assurances, mutuelles" if formel else basis["label"]) + " — cliquez une région ou une commune"
            card_title("account_balance", f"Carte territoriale — {basis['map_title']}", note)
            tmap.render(ds, f, key=f"sf_map_{f.density_basis}", value_col=basis["count_col"],
                        unit=basis["unit"], height=420, show_etab=basis["show_etab"],
                        hover_fields=basis["hover"], density_col=basis["density_col"],
                        density_title=basis["kpi_dens"], hide_empty=basis["count_col"])

    with visual_col:
        with st.container(key="card_sf_cat"):
            if f.density_basis == "mm":
                card_title("compare_arrows", "Répartition par opérateur, par région", "Moov, Togocom, duo ou non spécifié")
                table = M.mm_operator_by_region(ds, f)
                colors = {c: C.OPERATEUR_COLORS.get(c, C.OTHER_COLOR) for c in table.columns}
            elif f.density_basis == "points":
                card_title("category", "Répartition des points d'accès, par région",
                           "Banque · Micro-Finance · Assurance · Mutuelle · Agents MM")
                table = M.points_by_region(ds, f)
                colors = {c: C.ETAB_CATEGORY_COLORS.get(c, C.COLORS["yellow"]) for c in table.columns}
            else:
                card_title("category", "Répartition par catégorie, par région", "Banque · Micro-Finance · Assurance · Mutuelle")
                table = M.formel_category_by_region(ds, f)
                colors = {c: C.ETAB_CATEGORY_COLORS[c] for c in table.columns}
            table = M.restrict_regions_by_density(ds, f, table)
            if table.empty:
                empty_state(300)
            else:
                plot(ch.grouped_hbars(table, colors, height=360, stack=True), "sf_cat")

    with st.container(key="card_sf_ratio"):
        card_title("insights", "Accessibilité réelle",
                   f"{basis['kpi_hab']} — plus la barre est longue, moins le territoire est desservi")
        col_pref, col_reg = st.columns(2, gap="medium")
        with col_pref:
            st.caption("Par préfecture")
            table = M.basis_ratio_table(ds, f, "prefecture").dropna(subset=["hab"]) \
                .sort_values("hab", ascending=False).head(15).reset_index(drop=True)
            if table.empty:
                empty_state(340)
            else:
                bar = table.rename(columns={"prefecture": "label", "hab": "valeur"})[["label", "valeur"]]
                colors = [_CLASSE_COLORS.get(c, C.COLORS["muted"]) for c in table["classe"]] if formel \
                    else [C.COLORS["part"]] * len(bar)
                plot(ch.top_bars(bar, height=max(200, 24 * len(bar)), colors=colors, value_format=fmt_int,
                                 hover_extra=_ratio_hover(table, "prefecture")), "sf_ratio_pref")
                if formel:
                    st.markdown(_legend_html(), unsafe_allow_html=True)
        with col_reg:
            st.caption("Par région")
            table = M.basis_ratio_table(ds, f, "region").dropna(subset=["hab"]) \
                .sort_values("hab", ascending=False).head(20).reset_index(drop=True)
            if table.empty:
                empty_state(340)
            else:
                bar = table.rename(columns={"region_a": "label", "hab": "valeur"})[["label", "valeur"]]
                colors = [ch.region_color(l) for l in bar["label"]]
                plot(ch.radial_bars(bar, height=380, colors=colors, value_format=fmt_int,
                                    hover_extra=_ratio_hover(table, "region")), "sf_ratio_region")

    with st.container(key="card_sf_faible"):
        card_title("warning", "Territoires les moins bien desservis" if not formel else "Territoires à plus faible présence formelle",
                   f"Rayon = {basis['kpi_hab'].lower()} · plus long = moins bien desservi")
        bar = M.weakest_prefectures(ds, f)
        if bar.empty:
            empty_state(300)
        else:
            chart_df = bar.rename(columns={"prefecture": "label", "hab": "valeur"})
            chart_df = chart_df.assign(classe=chart_df["classe"] if formel else None).dropna(subset=["valeur"])
            if chart_df.empty:
                empty_state(300)
            else:
                colors = [_CLASSE_COLORS.get(c, C.COLORS["muted"]) for c in chart_df["classe"]] if formel \
                    else [C.COLORS["part"]] * len(chart_df)
                plot(ch.radial_bars(chart_df[["label", "valeur"]], height=420, colors=colors,
                                    value_format=fmt_int,
                                    hover_extra=_ratio_hover(bar.dropna(subset=["hab"]), "prefecture")),
                     "sf_faible")
                if formel:
                    st.markdown(_legend_html(), unsafe_allow_html=True)

    source_note(C.SOURCE_NOTE)


def _legend_html() -> str:
    items = "".join(
        f'<span style="display:inline-flex;align-items:center;margin-right:16px;font-size:12.5px;color:#334155">'
        f'<i style="background:{color};width:10px;height:10px;border-radius:3px;'
        f'display:inline-block;margin-right:6px"></i>{label}</span>'
        for label, color in _CLASSE_COLORS.items()
    )
    return f'<div style="margin-top:8px">{items}</div>'


def _ratio_hover(table: pd.DataFrame, level: str) -> list[str]:
    """Lignes d'info-bulle (population, établissements, taux, classe) — remplace le tableau
    « Détail complet » en intégrant directement ces informations aux graphiques."""
    lines = []
    for _, row in table.iterrows():
        parts = []
        if pd.notna(row.get("population")):
            parts.append(f"Population : {fmt_int(row['population'])}")
        if pd.notna(row.get("n")):
            parts.append(f"Effectif : {fmt_int(row['n'])}")
        if pd.notna(row.get("d10k")):
            parts.append(f"Pour 10 000 hab. : {fmt_dec(row['d10k'], 2)}")
        if level == "prefecture" and pd.notna(row.get("classe")):
            parts.append(f"Classe : {row['classe']}")
        lines.append("<br>".join(parts))
    return lines
