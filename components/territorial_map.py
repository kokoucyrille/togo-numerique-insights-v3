"""Carte territoriale interactive et synchronisée avec les filtres.

Seules les 5 régions disposent d'un contour administratif fiable (GeoJSON).
Aucune frontière de préfecture, de commune ou de canton n'est disponible ni
inventée : le zoom sur ces échelons s'appuie sur les coordonnées réelles des
communes (lon_c/lat_c) et, pour les établissements, sur leurs coordonnées
individuelles géolocalisées.

Un clic sur une région ou une commune écrit directement dans st.session_state
(mêmes clés que la barre latérale, cf. utils/filters.py) puis déclenche un
st.rerun() : la sélection se propage donc aux filtres, et réciproquement tout
changement de filtre re-zoome automatiquement la carte au prochain rendu.
"""
from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import config as C
from utils import geo
from utils import metrics as m
from utils.filters import F_CANTON, F_COMMUNE, F_GEO_NONCE, F_PREFECTURE, F_REGION, Filters
from utils.formatting import fmt_dec, fmt_int, fmt_pct

from .charts import FONT, MUTED, NAVY, _polygon_xy, _text_color, region_color

_DEFAULT_FORMAT = fmt_int

# --------------------------------------------------------------------------- #
# Champs additionnels d'info-bulle : chaque page choisit un sous-ensemble pertinent
# (hover_fields) parmi les colonnes déjà présentes dans les tables — aucune estimation,
# simple mise en forme. Un champ absent de la ligne (niveau région vs commune) ou vide
# (NaN) est silencieusement omis.
# --------------------------------------------------------------------------- #
_FIELD_META: dict[str, tuple[str, str]] = {
    "population": ("Population", "int"),
    "part_pop_pct": ("Part de la population nationale", "pct"),
    "n_formel": ("Établissements formels", "int"),
    "n_mm": ("Agents Mobile Money", "int"),
    "n_points": ("Points d'accès (formel + MM)", "int"),
    "formel_10k": ("Établissements / 10 000 hab.", "dec2"),
    "mm_10k": ("Agents MM / 10 000 hab.", "dec2"),
    "pts_10k": ("Points d'accès / 10 000 hab.", "dec2"),
    "part_formel_pct": ("Part des établissements du pays", "pct"),
    "hab_par_formel": ("Habitants par établissement", "int"),
    "hab_par_mm": ("Habitants par agent MM", "int"),
    "dist_med_km": ("Distance médiane au point formel", "km1"),
    "pct_eloignes": ("Agents à plus de 10 km d'un point formel", "pct"),
    "role": ("Rôle territorial", "text"),
    "niv_formel": ("Niveau de couverture formelle", "text"),
    "niv_mm": ("Niveau de couverture Mobile Money", "text"),
    "IB": ("Indice de besoin (IB)", "dec1"),
    "mm_only": ("Desservie uniquement par Mobile Money", "bool"),
}


def _fmt_field(kind: str, value) -> str:
    if kind == "int":
        return fmt_int(value)
    if kind == "dec2":
        return fmt_dec(value, 2)
    if kind == "dec1":
        return fmt_dec(value, 1)
    if kind == "pct":
        return fmt_pct(value, 1)
    if kind == "km1":
        return f"{fmt_dec(value, 1)} km"
    if kind == "bool":
        return "Oui" if value else "Non"
    return str(value)


def _hover_lines(row: pd.Series, fields: tuple[str, ...]) -> str:
    lines = []
    for col in fields:
        if col not in row.index:
            continue
        val = row[col]
        if pd.isna(val):
            continue
        label, kind = _FIELD_META.get(col, (col, "text"))
        lines.append(f"{label} : <b>{_fmt_field(kind, val)}</b>")
    return "<br>".join(lines)


def _pad(lon0: float, lon1: float, lat0: float, lat1: float, pad: float) -> tuple[float, float, float, float]:
    if lon1 - lon0 < C.MAP_MIN_SPAN:
        mid = (lon0 + lon1) / 2
        lon0, lon1 = mid - C.MAP_MIN_SPAN / 2, mid + C.MAP_MIN_SPAN / 2
    if lat1 - lat0 < C.MAP_MIN_SPAN:
        mid = (lat0 + lat1) / 2
        lat0, lat1 = mid - C.MAP_MIN_SPAN / 2, mid + C.MAP_MIN_SPAN / 2
    return lon0 - pad, lon1 + pad, lat0 - pad, lat1 + pad


def _bounds_for(communes: pd.DataFrame, f: Filters) -> tuple[tuple[float, float, float, float], str, str]:
    """Emprise réelle (lon0, lon1, lat0, lat1) du territoire le plus précis actuellement
    sélectionné, + son niveau ('commune'|'prefecture'|'region'|'national') et son libellé."""
    if f.commune:
        sub = communes[communes["label"].isin(f.commune)]
        if not sub.empty:
            b = (sub["lon_c"].min(), sub["lon_c"].max(), sub["lat_c"].min(), sub["lat_c"].max())
            return _pad(*b, C.MAP_PAD_COMMUNE), "commune", " · ".join(f.commune)
    if f.prefecture:
        sub = communes[communes["prefecture"].isin(f.prefecture)]
        if f.region:
            sub = sub[sub[m.REGION_COL].isin(f.region)]
        if not sub.empty:
            b = (sub["lon_c"].min(), sub["lon_c"].max(), sub["lat_c"].min(), sub["lat_c"].max())
            return _pad(*b, C.MAP_PAD_PREFECTURE), "prefecture", " · ".join(f.prefecture)
    if f.region:
        poly_bounds = geo.region_polygon_bounds()
        sub = communes[communes[m.REGION_COL].isin(f.region)]
        xs0, xs1, ys0, ys1 = [], [], [], []
        for r in f.region:
            if r in poly_bounds:
                x0, x1, y0, y1 = poly_bounds[r]
                xs0.append(x0); xs1.append(x1); ys0.append(y0); ys1.append(y1)
        if not sub.empty:
            xs0.append(sub["lon_c"].min()); xs1.append(sub["lon_c"].max())
            ys0.append(sub["lat_c"].min()); ys1.append(sub["lat_c"].max())
        if xs0:
            b = (min(xs0), max(xs1), min(ys0), max(ys1))
            return _pad(*b, C.MAP_PAD_REGION), "region", " · ".join(f.region)
    b = geo.national_bounds()
    return _pad(*b, C.MAP_PAD_NATIONAL), "national", "Togo"


_DENSITY_STOPS = ("#E3F4EC", "#5FCFA0", "#0E9F6E", "#02403A")


def _density_color(value, lo: float, hi: float) -> str | None:
    """Couleur (dégradé vert clair → foncé) d'une densité ; None si la valeur est absente."""
    if value is None or pd.isna(value):
        return None
    t = min(max((float(value) - lo) / (hi - lo), 0.0), 1.0) if hi > lo else 0.0
    pos = t * (len(_DENSITY_STOPS) - 1)
    i = min(int(pos), len(_DENSITY_STOPS) - 2)
    a, b = _DENSITY_STOPS[i], _DENSITY_STOPS[i + 1]
    frac = pos - i
    mix = [round(int(a[k:k + 2], 16) + (int(b[k:k + 2], 16) - int(a[k:k + 2], 16)) * frac) for k in (1, 3, 5)]
    return "#{:02X}{:02X}{:02X}".format(*mix)


def _commune_color(row: pd.Series, color_by: str | None, vmax: float, base_color: str) -> str:
    if color_by == "palier" and pd.notna(row.get("palier")):
        return C.PALIER_COLORS.get(row["palier"], base_color)
    return base_color


def build_figure(ds, f: Filters, *, value_col: str = "n_points", unit: str = "", value_format=_DEFAULT_FORMAT,
                 color_by: str | None = None, height: int = 480, show_etab: bool = False,
                 hover_fields: tuple[str, ...] = (), density_col: str | None = None,
                 density_title: str = "") -> tuple[go.Figure, str, str]:
    regions_df = m.region_values(ds, f, value_col)
    values = dict(zip(regions_df["label"], regions_df["valeur"])) if not regions_df.empty else {}
    dens_vals: dict = {}
    d_lo, d_hi = 0.0, 1.0
    if density_col:
        # Coloration par densité (pour 10 000 hab.) : régions et communes partagent une même
        # échelle, bornée au 5e–95e centile des communes visibles pour rester lisible.
        rd = m.region_values(ds, f, density_col)
        dens_vals = dict(zip(rd["label"], rd["valeur"])) if not rd.empty else {}
    region_detail = m.region_hover_detail(ds, f) if hover_fields else pd.DataFrame()
    communes = m.commune_points(ds, f, value_col)
    if density_col and density_col in communes.columns and communes[density_col].notna().any():
        d_lo = float(communes[density_col].quantile(0.05))
        d_hi = float(communes[density_col].quantile(0.95))
        if d_hi <= d_lo:
            d_lo, d_hi = float(communes[density_col].min()), float(communes[density_col].max())
    bounds, level, label = _bounds_for(communes, f)
    x0, x1, y0, y1 = bounds

    geo_json = geo.load_regions_geojson()
    fig = go.Figure()
    lons: list[float] = []
    lats: list[float] = []
    poly_names: set[str] = set()

    for i, feature in enumerate(geo_json["features"]):
        props = feature["properties"]
        data_name = C.REGION_GEO_ALIAS.get(props["region"], props["region"])
        poly_names.add(data_name)
        has = data_name in values
        if f.density_range is not None and not has:
            continue  # région hors plage de densité : n'apparaît pas sur la carte
        selected = data_name in f.region
        color = region_color(data_name, i) if (has or selected or not values) else C.COLORS["empty"]
        if f.region and not selected:
            color = C.COLORS["empty"]
        elif density_col and has and (dc := _density_color(dens_vals.get(data_name), d_lo, d_hi)):
            color = dc
        xs, ys = _polygon_xy(feature["geometry"])
        lons += [x for x in xs if x is not None]
        lats += [y for y in ys if y is not None]
        hover = f"<b>{data_name}</b>" + (f"<br>{value_format(values[data_name])} {unit}" if has else "")
        if has and hover_fields and data_name in region_detail.index:
            extra_lines = _hover_lines(region_detail.loc[data_name], hover_fields)
            if extra_lines:
                hover += "<br>" + extra_lines
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", fill="toself", fillcolor=color, name=data_name,
            line=dict(color="#FFFFFF", width=1.6), hoveron="fills", text=hover, hoverinfo="text",
            showlegend=False, customdata=[["region", data_name]] * len(xs),
        ))
        fig.add_trace(go.Scatter(
            # Marqueur volontairement quasi invisible : sert uniquement de zone cliquable
            # fiable pour la sélection (les traces "lines" du polygone ne sont pas
            # sélectionnables par clic en Plotly ; seuls les points à marqueur le sont).
            x=[props["cx"]], y=[props["cy"]], mode="markers+text",
            marker=dict(size=68, color=color, opacity=0.015, line=dict(width=0)),
            text=[f"{data_name}<br><b>{value_format(values[data_name])}</b>" if has else data_name],
            textposition="middle center", showlegend=False,
            textfont=dict(family=FONT, size=11, color=_text_color(color) if (has or selected) else MUTED),
            customdata=[["region", data_name]], hovertemplate=hover + "<extra></extra>",
        ))

    point_regions = [(n, v) for n, v in values.items() if n not in poly_names and n in C.GEO_POINTS]
    for name, value in point_regions:
        lat, lon = C.GEO_POINTS[name]
        selected = name in f.region
        color = region_color(name) if (selected or not f.region) else C.COLORS["empty"]
        if density_col and (selected or not f.region) and (dc := _density_color(dens_vals.get(name), d_lo, d_hi)):
            color = dc
        point_hover = f"<b>{name}</b><br>{value_format(value)} {unit}"
        if hover_fields and name in region_detail.index:
            extra_lines = _hover_lines(region_detail.loc[name], hover_fields)
            if extra_lines:
                point_hover += "<br>" + extra_lines
        fig.add_trace(go.Scatter(
            x=[lon], y=[lat], mode="markers+text", showlegend=False,
            marker=dict(size=16, color=color, line=dict(color="#FFFFFF", width=1.5)),
            text=[f"{name}<br><b>{value_format(value)}</b>"], textposition="bottom center",
            textfont=dict(family=FONT, size=11, color=NAVY),
            customdata=[["region", name]],
            hovertemplate=point_hover + "<extra></extra>",
        ))

    if not communes.empty:
        vmax = communes["valeur"].max() or 1
        base_color = C.COLORS["part"]
        in_scope = communes[communes["dans_perimetre"]]
        out_scope = communes[~communes["dans_perimetre"]]
        for subset, opacity, size_boost in ((out_scope, 0.28, 0.6), (in_scope, 0.92, 1.0)):
            if subset.empty:
                continue
            sizes = [7 + size_boost * 13 * math.sqrt(max(v, 0) / vmax) for v in subset["valeur"]]
            colors = [(_density_color(r.get(density_col), d_lo, d_hi) if density_col else None)
                      or _commune_color(r, color_by, vmax, base_color) for _, r in subset.iterrows()]
            selected_mask = subset["label"].isin(f.commune)
            line_widths = [2.4 if sel else 0.8 for sel in selected_mask]
            line_colors = ["#0B1F33" if sel else "#FFFFFF" for sel in selected_mask]
            hover = [f"<b>{r.label}</b> ({r.prefecture})<br>{value_format(r.valeur)} {unit}"
                    + (f"<br>{C.PALIER_LABELS[r.palier].split(' — ', 1)[1].capitalize()}"
                       if pd.notna(r.get('palier')) and r.palier in C.PALIER_LABELS else "")
                    + (f"<br>{extra_lines}" if hover_fields and (extra_lines := _hover_lines(r, hover_fields)) else "")
                    for _, r in subset.iterrows()]
            fig.add_trace(go.Scatter(
                x=subset["lon_c"], y=subset["lat_c"], mode="markers", showlegend=False,
                marker=dict(size=sizes, color=colors, opacity=opacity,
                           line=dict(color=line_colors, width=line_widths)),
                customdata=[["commune", v] for v in subset["label"]],
                text=hover, hoverinfo="text",
            ))

    if show_etab and level in ("prefecture", "commune"):
        pts = m.etab_points(ds, f)
        if not pts.empty:
            colors = [C.ETAB_CATEGORY_COLORS.get(c, C.OTHER_COLOR) for c in pts["activite_categorie"]]
            hover = [f"<b>{r.etab_nom}</b><br>{r.activite_categorie} — {r.commune_norm}"
                    + (f"<br>{r.nom_localite.strip()}"
                       if pd.notna(r.get("nom_localite")) and str(r.nom_localite).strip()
                       and str(r.nom_localite).strip() != r.commune_norm else "")
                    + (f"<br>{r.prefecture_norm} · {r.region_a}"
                       if pd.notna(r.get("prefecture_norm")) else "")
                    for _, r in pts.iterrows()]
            fig.add_trace(go.Scatter(
                x=pts["longitude"], y=pts["latitude"], mode="markers", showlegend=False,
                marker=dict(size=8, color=colors, symbol="diamond", line=dict(color="#FFFFFF", width=1)),
                text=hover, hoverinfo="text", customdata=[["etab", "-"]] * len(pts),
            ))

    if density_col:
        # Trace « fantôme » (aucun point) qui ne sert qu'à afficher la légende de l'échelle.
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers", showlegend=False, hoverinfo="skip",
            marker=dict(color=[d_lo], cmin=d_lo, cmax=d_hi, showscale=True,
                        colorscale=[[i / (len(_DENSITY_STOPS) - 1), c] for i, c in enumerate(_DENSITY_STOPS)],
                        colorbar=dict(title=dict(text=density_title, side="right", font=dict(size=11)),
                                      thickness=10, len=0.55, x=1.0, xanchor="right", y=0.3,
                                      tickfont=dict(size=10), outlinewidth=0)),
        ))

    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=12, color=C.COLORS["slate"]),
        showlegend=False, dragmode=False, clickmode="event+select",
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=C.COLORS["grid"],
                       font=dict(family=FONT, size=12, color=NAVY)),
        transition=dict(duration=350, easing="cubic-in-out"),
    )
    fig.update_xaxes(visible=False, range=[x0, x1], constrain="domain", fixedrange=True)
    fig.update_yaxes(visible=False, range=[y0, y1], scaleanchor="x", scaleratio=1,
                     constrain="domain", fixedrange=True)
    return fig, level, label


_LEVEL_LABEL = {"national": "Vue nationale", "region": "Région", "prefecture": "Préfecture", "commune": "Commune"}


def _breadcrumb(f: Filters, level: str) -> str:
    parts = ['<span class="mapbc__item mapbc__item--home">Togo</span>']
    for values, cls in ((f.region, "region"), (f.prefecture, "prefecture"), (f.commune, "commune")):
        if values:
            parts.append(f'<span class="mapbc__sep">›</span><span class="mapbc__item mapbc__item--{cls}">'
                        f'{", ".join(values)}</span>')
    return '<div class="mapbc">' + "".join(parts) + "</div>"


def _reset_geo() -> None:
    for key in (F_REGION, F_PREFECTURE, F_COMMUNE, F_CANTON):
        st.session_state.pop(key, None)
    # Force le remontage du widget st.plotly_chart (cf. utils/filters.py) : sans ça, la
    # sélection déjà cliquée sur la carte reste affichée malgré la réinitialisation.
    st.session_state[F_GEO_NONCE] = st.session_state.get(F_GEO_NONCE, 0) + 1


def _apply_map_selection(ds, key: str) -> None:
    event = st.session_state.get(key)
    points = (event.get("selection") or {}).get("points", []) if event else []
    if not points:
        return
    custom = points[0].get("customdata")
    if not custom or len(custom) != 2:
        return

    kind, value = custom
    if kind == "region":
        current = set(st.session_state.get(F_REGION, ()))
        if value in current:
            current.discard(value)
        elif len(current) < C.MAX_COMPARE:
            current.add(value)
        else:
            current = {value}
        st.session_state[F_REGION] = tuple(sorted(current))
        for state_key in (F_PREFECTURE, F_COMMUNE, F_CANTON):
            st.session_state.pop(state_key, None)
    elif kind == "commune":
        communes_all = ds.get("table_analytique_communes")
        if communes_all.empty:
            return
        row = communes_all.loc[communes_all["commune"] == value]
        if row.empty:
            return
        st.session_state[F_REGION] = (str(row.iloc[0][m.REGION_COL]),)
        st.session_state[F_PREFECTURE] = (str(row.iloc[0]["prefecture"]),)
        st.session_state[F_COMMUNE] = (value,)
        st.session_state.pop(F_CANTON, None)
    else:
        return
    # Les multiselect de la barre latérale (utils/filters.py:_multi) sont rendus
    # sous une clé physique suffixée par ce même compteur, et ne se resynchronisent
    # avec la clé « logique » (F_REGION, etc.) que lorsque cette clé physique est
    # neuve. Sans cet incrément, un clic sur la carte mettrait à jour l'état mais
    # les cases cochées dans la barre latérale ne suivraient pas visuellement.
    st.session_state[F_GEO_NONCE] = st.session_state.get(F_GEO_NONCE, 0) + 1


def render(ds, f: Filters, *, key: str, value_col: str = "n_points", unit: str = "points",
          value_format=_DEFAULT_FORMAT, color_by: str | None = None, height: int = 480,
          show_etab: bool = False, caption: str | None = None,
          hover_fields: tuple[str, ...] = (), density_col: str | None = None,
          density_title: str = "") -> None:
    """Carte interactive : clic direct sur une région ou une commune, zoom automatique
    sur le territoire réellement sélectionné (filtres ou clic), synchronisée partout."""
    fig, level, _ = build_figure(ds, f, value_col=value_col, unit=unit, value_format=value_format,
                                 color_by=color_by, height=height, show_etab=show_etab,
                                 hover_fields=hover_fields, density_col=density_col,
                                 density_title=density_title)

    top = st.columns([5, 1.2])
    with top[0]:
        st.markdown(_breadcrumb(f, level), unsafe_allow_html=True)
    with top[1]:
        st.button("Vue nationale", icon=":material/public:", key=f"{key}_reset",
                  on_click=_reset_geo, disabled=not f.has_geo, use_container_width=True, type="tertiary")

    plot_config = {"displayModeBar": False, "scrollZoom": False, "responsive": True}
    # La clé du widget intègre le compteur de réinitialisation géographique : Streamlit
    # garde la sélection d'un st.plotly_chart tant que sa clé ne change pas, donc sans ce
    # suffixe un clic sur « Réinitialiser » (barre latérale ou bouton ci-dessus) effacerait
    # bien les filtres mais laisserait la carte visuellement figée sur l'ancienne sélection.
    chart_key = f"{key}_{st.session_state.get(F_GEO_NONCE, 0)}"
    on_select = lambda: _apply_map_selection(ds, chart_key)
    try:
        st.plotly_chart(fig, key=chart_key, on_select=on_select, selection_mode=["points"],
                                width="stretch", config=plot_config)
    except TypeError:
        st.plotly_chart(fig, key=chart_key, on_select=on_select, selection_mode=["points"],
                                use_container_width=True, config=plot_config)

    if caption:
        st.caption(caption)
