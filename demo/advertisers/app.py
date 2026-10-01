"""Advertisers – YoY overview (public demo).

An anonymised imitation of an advertiser-spend app I built end to end at work:
weekly spend of advertisers in search compared with the same weeks last year.
The layout follows the real app; advertisers are numbered, and every amount,
search and trend is SIMULATED at start-up. Nothing comes from real data.

Run locally:  streamlit run app.py
In the browser it runs through stlite (see index.html).
"""
import html
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Advertisers – YoY overview · demo", layout="wide")

WEEKS = 52
THIS, LAST = 2026, 2025
MONDAY = {THIS: date(2026, 1, 5), LAST: date(2025, 1, 6)}
VERTICALS = ["Retail", "Travel", "Auto", "Finance", "Real estate", "Services"]
PEAK = {"Retail": 47, "Travel": 22, "Auto": 14, "Finance": 4, "Real estate": 12, "Services": 36}
BLUE, GREY, GREEN, RED = "#0063e8", "#9ca3af", "#1a7f37", "#c9252c"
NBSP = " "

CSS = """
.stApp, [data-testid="stMain"], header[data-testid="stHeader"] { background: #f0f0f0; }
[data-testid="stSidebar"] [data-testid="stTextInputRootElement"], [data-testid="stSidebar"] div[data-baseweb="input"], [data-testid="stSidebar"] div[data-baseweb="select"] > div { border: 1px solid #dfdfdf; border-radius: 4px; background: #fff; }
[data-testid="stSidebar"] { background: #fff; border-right: 1px solid #dfdfdf; }
html, body, [class*="css"], .stMarkdown, button, input { font-family: "Helvetica Neue", Arial, sans-serif; }
[data-testid="stMainBlockContainer"] { padding-top: 2.4rem; max-width: 1000px; }
div[class*="st-key-box"] { background: #fff; border: 1px solid #dfdfdf; border-radius: 4px; padding: 22px 24px; }
div[class*="st-key-card"] { background: #fff; border: 1px solid #dfdfdf; border-radius: 4px; padding: 14px 16px; }
.dash-title { font-size: 26px; font-weight: 700; color: #272727; margin: 0 0 6px; }
.meta { color: #555; font-size: 12.5px; line-height: 1.6; }
.badge { display: inline-block; margin-top: 10px; background: #fff4e5; border: 1px solid #f3c48b; border-radius: 4px; padding: 2px 8px; font-size: 12.5px; color: #555; }
.badge b { color: #272727; }
.sec-title { font-size: 20px; font-weight: 700; color: #272727; margin: 0; }
.sec-note { color: #555; font-size: 14.5px; margin: 6px 0 0; max-width: 70ch; }
.card-title { font-weight: 700; font-size: 14.5px; color: #272727; text-decoration: underline dotted #999; text-underline-offset: 4px; }
.cmp { display: flex; align-items: flex-end; gap: 22px; margin: 14px 0 10px; flex-wrap: wrap; }
.cmp .y { font-size: 12px; color: #555; text-decoration: underline dotted #aaa; text-underline-offset: 3px; }
.cmp .old { font-size: 17px; color: #555; margin-top: 2px; }
.cmp .new { font-size: 20px; font-weight: 700; color: #272727; margin-top: 2px; }
.chip { display: inline-block; font-size: 13px; font-weight: 700; padding: 5px 10px; border-radius: 3px; white-space: nowrap; }
.chip.up { background: #e3f4e6; color: #1a7f37; } .chip.down { background: #fde7e7; color: #c9252c; } .chip.flat { background: #eee; color: #555; }
.chip.sm { font-size: 11px; padding: 2px 6px; font-weight: 600; }
.foot { border-top: 1px solid #eee; padding-top: 8px; font-size: 12.5px; color: #555; }
.foot b.up { color: #1a7f37; } .foot b.down { color: #c9252c; }
.row-meta { font-size: 12px; color: #555; margin: -6px 0 4px; }
.row-meta .chip { margin-right: 4px; }
.side-h { font-size: 12.5px; font-weight: 700; color: #555; margin: 8px 0 0; }
div[class*="st-key-row_"] { border-bottom: 1px solid #eee; padding: 8px 10px; margin: 0 -10px; gap: 2px; }
[data-testid="stSidebar"] [data-testid="stVerticalBlock"]:has(> div[class*="st-key-row_"]) { gap: 0; }
div[class*="st-key-row_"] button { padding: 0; min-height: 0; font-weight: 700; color: #272727; }
div[class*="st-key-row_"] button p { font-size: 13.5px; }
div[data-testid="stButtonGroup"] button { border-radius: 4px; border: 1px solid #dfdfdf; background: #fff; }
div[data-testid="stButtonGroup"] button[kind$="Active"] { background: #e6f0fd; border-color: #0063e8; color: #0063e8; }
div.st-key-nav div[data-testid="stButtonGroup"] button { border: 0; padding: 6px 14px; }
div.st-key-nav div[data-testid="stButtonGroup"] button[kind$="Active"] { background: #e6f0fd; }
"""


# =====================
# SYNTHETIC MARKET
# =====================
@st.cache_data
def simulate(seed: int, n: int = 400):
    rng = np.random.default_rng(seed)
    names = [f"Advertiser {i:03d}" for i in range(1, n + 1)]
    vertical = rng.choice(VERTICALS, n, p=[.3, .14, .12, .14, .1, .2])
    base = (rng.pareto(1.6, n) + 1) * 4000                       # weekly spend, heavy-tailed
    growth = rng.lognormal(0, .28, n)
    start = np.where(rng.random(n) < .08, rng.integers(1, 30, n), 0)
    stop = np.where(rng.random(n) < .07, rng.integers(5, 35, n), 99)
    peak = np.array([PEAK[v] for v in vertical])
    w = np.arange(1, WEEKS + 1)
    season = 1 + .35 * np.cos(2 * np.pi * (w[None, :] - peak[:, None]) / WEEKS)
    last = base[:, None] * season * rng.lognormal(0, .18, (n, WEEKS))
    last[start > 0] = 0                                           # newcomers had no spend last year
    this = base[:, None] * season * rng.lognormal(0, .18, (n, WEEKS)) * growth[:, None]
    this[w[None, :] < start[:, None]] = 0
    this[w[None, :] >= stop[:, None]] = 0
    cpr = rng.uniform(.35, .9, n)                                 # money per search the advertiser appears in
    frames = []
    for year, m in [(LAST, last), (THIS, this)]:
        frames.append(pd.DataFrame({
            "advertiser": np.repeat(names, WEEKS), "vertical": np.repeat(vertical, WEEKS),
            "year": year, "week": np.tile(w, n), "spend": m.ravel().round(),
            "searches": (m / cpr[:, None]).ravel().round()}))
    df = pd.concat(frames, ignore_index=True)
    mseason = 1 + .12 * np.cos(2 * np.pi * (w - 48) / WEEKS)
    market = pd.DataFrame({"year": np.repeat([LAST, THIS], WEEKS), "week": np.tile(w, 2),
                           "searches": np.r_[mseason * 1.6e6, mseason * 1.6e6 * .965] * rng.lognormal(0, .02, 2 * WEEKS)})
    return df, market


def czk(v):
    s = f"{abs(round(v)):,}".replace(",", NBSP)
    return f"{'-' if v < 0 else ''}{s}{NBSP}Kč"


def czk_signed(v):
    return ("+" if v >= 0 else "") + czk(v)


def chg(new, old):
    if old == 0:
        return ("new", "up") if new else ("–", "flat")
    if new == 0:
        return ("no spend", "down")
    r = new / old - 1
    if abs(r) < .0005:
        return ("0.0 %", "flat")
    return (f"{'↑' if r >= 0 else '↓'} {100 * r:+.1f} %", "up" if r >= 0 else "down")


def chip(new, old, cls="", prefix=""):
    text, d = chg(new, old)
    return f"<span class='chip {d} {cls}'>{prefix}{text}</span>"


def week_dates(year, weeks):
    a = MONDAY[year] + timedelta(weeks=weeks[0] - 1)
    b = MONDAY[year] + timedelta(weeks=weeks[-1] - 1, days=6)
    return f"{a:%d.%m.%Y} – {b:%d.%m.%Y}"


def style(fig, h=380):
    fig.update_layout(template="plotly_white", height=h, margin=dict(l=50, r=10, t=30, b=40),
                      font=dict(family="Helvetica Neue, Arial, sans-serif", size=12, color="#272727"),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title_text=""))
    return fig


def md(text):
    st.markdown(text, unsafe_allow_html=True)


# =====================
# STATE + SIDEBAR
# =====================
md(f"<style>{CSS}</style>")
ss = st.session_state
ss.setdefault("pick", None)


@st.dialog("About the app", width="large")
def about():
    st.markdown(
        "A **public demo** of an advertiser-spend app I built at work, from the first analysis to the deployed "
        "app. It compares every advertiser's weekly spend in search with the same weeks last year.\n\n"
        "**Everything is simulated.** 400 numbered advertisers in six made-up verticals, with seasonality, "
        "growth, newcomers and leavers; the market's searches are simulated too.\n\n"
        "#### How it is computed\n"
        "- **Week by position.** Week N this year is compared with week N last year, not by date.\n"
        "- **Closed weeks only.** The last week of the data is treated as incomplete.\n"
        "- **YTD / 4W / YTD − 4W.** Sum over closed weeks, the last four of them, and the rest.\n"
        "- **Cohort.** The biggest advertisers *by this year's spend*. This favours those who grew, so a "
        "cohort's YoY looks better than the market's.\n"
        "- **Inventory vs advertising.** Inventory = YoY change in the number of searches; advertising = "
        "advertiser YoY minus inventory.")


def pick(name):
    ss.pick = name


with st.sidebar:
    if st.button("About the app", width="stretch"):
        about()
    with st.expander("Filters"):
        verticals = st.multiselect("Vertical", VERTICALS)
        as_of = st.slider("Closed weeks this year", 8, 51, 37)
        seed = st.number_input("Synthetic market", 1, 50, 4)
    data, market = simulate(int(seed))
    if verticals:
        data = data[data.vertical.isin(verticals)]
    ytd_w, w4, rest = list(range(1, as_of + 1)), list(range(as_of - 3, as_of + 1)), list(range(1, as_of - 3))
    with st.expander("Data and export"):
        st.caption(f"Simulated weekly spend, weeks 1–{as_of} of {THIS} and all of {LAST}.")
        st.download_button("Download CSV", data[data.week.isin(ytd_w)].to_csv(index=False), "advertisers_demo.csv",
                           "text/csv", width="stretch")

    tot = data[data.week.isin(ytd_w)].pivot_table(index="advertiser", columns="year", values="spend", aggfunc="sum",
                                                   fill_value=0).reindex(columns=[LAST, THIS], fill_value=0)
    t4 = data[data.week.isin(w4)].pivot_table(index="advertiser", columns="year", values="spend", aggfunc="sum",
                                               fill_value=0).reindex(columns=[LAST, THIS], fill_value=0)
    market_ytd = tot[THIS].sum()
    md(f"<div class='side-h'>Ranking by spend {THIS}</div>")
    query = st.text_input("Search", placeholder="Find an advertiser…", label_visibility="collapsed")
    order = st.selectbox("Order", ["Ranking (spend)", "Biggest YTD growth", "Biggest YTD drop"],
                         label_visibility="collapsed")
    ranked = tot.sort_values(THIS, ascending=False)
    rank_of = {a: i + 1 for i, a in enumerate(ranked.index)}
    yoy = tot[THIS] / tot[LAST].replace(0, np.nan) - 1
    if order == "Biggest YTD growth":
        ranked = ranked.loc[yoy.reindex(ranked.index).sort_values(ascending=False).index]
    elif order == "Biggest YTD drop":
        ranked = ranked.loc[yoy.reindex(ranked.index).sort_values().index]
    if query:
        ranked = ranked[ranked.index.str.contains(query, case=False)]
    st.caption(f"{min(len(ranked), 30)} of {len(tot)} advertisers")
    if ss.pick not in tot.index:
        ss.pick = tot[THIS].idxmax()
    shown = list(ranked.index[:30])
    if ss.pick in shown:
        sel = shown.index(ss.pick)
        md(f"<style>div.st-key-row_{sel} {{ background: #e6f0fd; }} div.st-key-row_{sel} button p {{ color: #0063e8; }}</style>")
    for i, name in enumerate(shown):
        with st.container(key=f"row_{i}"):
            st.button(f"{rank_of[name]}. {name}", key=f"pick_{i}", type="tertiary", on_click=pick, args=(name,))
            md(f"<div class='row-meta'>{czk(tot.loc[name, THIS])} · {100 * tot.loc[name, THIS] / market_ytd:.1f} % of money<br>"
               f"{chip(tot.loc[name, THIS], tot.loc[name, LAST], 'sm', 'YTD ')}"
               f"{chip(t4.loc[name, THIS], t4.loc[name, LAST], 'sm', '4W ')}</div>")


# =====================
# HEADER
# =====================
def window_sum(weeks, year, frame=None):
    f = data if frame is None else frame
    return f[(f.year == year) & f.week.isin(weeks)].spend.sum()


with st.container(key="box_header"):
    md(f"<div class='dash-title'>Advertisers – YoY overview</div>"
       f"<div class='meta'>{len(tot)} advertisers · simulated market<br>"
       f"Comparison: YTD (weeks 1–{as_of}): {week_dates(THIS, ytd_w)} vs {week_dates(LAST, ytd_w)} · "
       f"4W (weeks {w4[0]}–{w4[-1]}): {week_dates(THIS, w4)} vs {week_dates(LAST, w4)}</div>"
       f"<div class='badge'>data <b>{MONDAY[LAST]:%d.%m.%Y} – {week_dates(THIS, [as_of]).split(' – ')[1]} · simulated</b></div>")
    with st.container(key="nav"):
        nav = st.segmented_control("Section", ["Overview", "Market change", "Concentration", "Outlook", "Signals"],
                                   default="Overview", label_visibility="collapsed")
nav = nav or "Overview"


def window_card(key, title, weeks, frame, note_market):
    old, new = window_sum(weeks, LAST, frame), window_sum(weeks, THIS, frame)
    mkt = window_sum(weeks, THIS)
    diff = new - old
    share = f" ({100 * diff / mkt:+.1f} % of market)" if note_market and mkt else ""
    with st.container(key=key):
        md(f"<div class='card-title'>{title}</div>"
           f"<div class='cmp'><div><div class='y'>{LAST}</div><div class='old'>{czk(old)}</div></div>"
           f"<div><div class='y'>{THIS}</div><div class='new'>{czk(new)}</div></div>{chip(new, old)}</div>"
           f"<div class='foot'><b class='{'up' if diff >= 0 else 'down'}'>{czk_signed(diff)}</b>{share}</div>")


def plain_card(key, title, cells, extra=""):
    body = "".join(f"<div><div class='y'>{k}</div><div class='new'>{v}</div></div>" for k, v in cells)
    with st.container(key=key):
        md(f"<div class='card-title' style='text-decoration:none'>{title}</div><div class='cmp'>{body}{extra}</div>")


# =====================
# OVERVIEW
# =====================
if nav == "Overview":
    with st.container(key="box_controls"):
        c1, c2, c3 = st.columns([1.2, 1.6, 1])
        view = c1.segmented_control("View", ["Advertiser", "Sum of cohort"], default="Advertiser")
        cohort_n = c2.segmented_control("Cohort", ["Top 20", "Top 100", "All"], default="Top 100")
        gran = c3.segmented_control("Granularity", ["Weeks", "Months"], default="Weeks")
    view, cohort_n, gran = view or "Advertiser", cohort_n or "Top 100", gran or "Weeks"
    n = {"Top 20": 20, "Top 100": 100, "All": len(tot)}[cohort_n]
    cohort = tot[THIS].nlargest(n).index
    if view == "Advertiser":
        frame, title = data[data.advertiser == ss.pick], ss.pick
        subtitle = f"Rank {rank_of[ss.pick]} by this year's spend · {data.loc[data.advertiser == ss.pick, 'vertical'].iat[0]}"
    else:
        frame, title = data[data.advertiser.isin(cohort)], f"Sum of {cohort_n.lower()}"
        subtitle = f"{len(cohort)} advertisers, picked by this year's spend"

    with st.container(key="box_main"):
        md(f"<div class='sec-title'>{html.escape(title)}</div><div class='sec-note'>{subtitle}</div>")
        a, b = st.columns(2)
        with a:
            window_card("card_ytd", "YTD", ytd_w, frame, True)
            window_card("card_rest", "YTD – 4W", rest, frame, True)
        with b:
            window_card("card_4w", "4W", w4, frame, True)
            s_old = frame[(frame.year == LAST) & frame.week.isin(ytd_w)]
            s_new = frame[(frame.year == THIS) & frame.week.isin(ytd_w)]
            y_old = s_old.spend.sum() / max(s_old.searches.sum(), 1)
            y_new = s_new.spend.sum() / max(s_new.searches.sum(), 1)
            plain_card("card_share", "Share and yield",
                       [("Market share YTD", f"{100 * s_new.spend.sum() / max(market_ytd, 1):.1f} %"),
                        ("Revenue per search", f"{y_old:.2f} Kč → {y_new:.2f} Kč")], chip(y_new, y_old, "sm"))
        mk = market[market.week.isin(ytd_w)].groupby("year").searches.sum()
        inv = mk[THIS] / mk[LAST] - 1
        adv_yoy = s_new.spend.sum() / max(s_old.spend.sum(), 1) - 1
        md(f"<div class='sec-note'>Inventory vs advertising: searches in the market changed by <b>{100 * inv:+.1f} %</b>, "
           f"so <b>{100 * (adv_yoy - inv):+.1f} pp</b> of the YTD change comes from advertising itself.</div>")

    with st.container(key="box_chart"):
        md("<div class='sec-title'>Spend over time</div>"
           f"<div class='sec-note'>Grey is {LAST}, blue {THIS}; granularity switches above.</div>")
        series = frame.groupby(["year", "week"]).spend.sum().unstack(0).reindex(columns=[LAST, THIS]).fillna(0)
        x = pd.to_datetime([MONDAY[LAST] + timedelta(weeks=int(w) - 1) for w in series.index])
        last_s = pd.Series(series[LAST].values, index=x)
        this_s = pd.Series(series[THIS].where(series.index <= as_of).values, index=x)
        if gran == "Months":
            closed_to = pd.Timestamp(MONDAY[LAST] + timedelta(weeks=as_of - 1, days=6))
            last_s, this_s = last_s.resample("MS").sum(), this_s.resample("MS").sum(min_count=1)
            this_s[this_s.index + pd.offsets.MonthEnd(0) > closed_to] = np.nan   # closed months only
        fig = go.Figure()
        fig.add_scatter(x=last_s.index, y=last_s.values, name=f"{LAST} (full year)", mode="lines+markers",
                        line=dict(color=GREY, width=1.5), marker=dict(size=4), fill="tozeroy",
                        fillcolor="rgba(156,163,175,.15)")
        fig.add_scatter(x=this_s.index, y=this_s.values, name=f"{THIS} (YTD)", mode="lines+markers",
                        line=dict(color=BLUE, width=2.5), marker=dict(size=4), fill="tozeroy",
                        fillcolor="rgba(0,99,232,.10)")
        fig.update_xaxes(tickformat="%d.%m.%Y", tickangle=-45)
        fig.update_yaxes(ticksuffix=" Kč", separatethousands=True)
        st.plotly_chart(style(fig), width="stretch")
        if st.toggle("Search volume under the chart"):
            sv = frame.groupby(["year", "week"]).searches.sum().unstack(0).reindex(columns=[LAST, THIS]).fillna(0)
            f2 = go.Figure([go.Bar(x=sv.index, y=sv[LAST], name=str(LAST), marker_color=GREY),
                            go.Bar(x=sv.index, y=sv[THIS].where(sv.index <= as_of), name=str(THIS), marker_color=BLUE)])
            f2.update_xaxes(title="week")
            st.plotly_chart(style(f2, 240), width="stretch")
        st.download_button("Download chart data (CSV)", series.to_csv(), "spend_over_time.csv", "text/csv")

# =====================
# MARKET CHANGE
# =====================
elif nav == "Market change":
    with st.container(key="box_controls"):
        win = st.segmented_control("Window", ["YTD", "4W", "YTD – 4W"], default="YTD") or "YTD"
    weeks = {"YTD": ytd_w, "4W": w4, "YTD – 4W": rest}[win]
    per = data[data.week.isin(weeks)].pivot_table(index="advertiser", columns="year", values="spend",
                                                    aggfunc="sum", fill_value=0).reindex(columns=[LAST, THIS], fill_value=0)
    per["delta"] = per[THIS] - per[LAST]
    gains, losses = per[per.delta > 0], per[per.delta < 0]
    with st.container(key="box_moved"):
        md(f"<div class='sec-title'>Who moved the market — {win}</div>"
           f"<div class='sec-note'>Computed over <b>all advertisers</b> in the filters ({len(per)}), not just a cohort; "
           "otherwise advertisers who stopped or started spending would be missing. Gains and losses add up exactly to "
           "the total change.</div>")
        a, b = st.columns(2)
        with a:
            window_card("card_mkt", f"Market change {win}", weeks, None, False)
            share5 = losses.delta.nsmallest(5).sum() / losses.delta.sum() if len(losses) else 0
            plain_card("card_loss", "Losses", [("Total", czk(losses.delta.sum())), ("Top 5 of it", f"{100 * share5:.1f} %")])
        with b:
            share5 = gains.delta.nlargest(5).sum() / gains.delta.sum() if len(gains) else 0
            plain_card("card_gain", "Gains", [("Total", czk(gains.delta.sum())), ("Top 5 of it", f"{100 * share5:.1f} %")])
            plain_card("card_net", "Net change", [("Difference", czk(per.delta.sum()))])
    with st.container(key="box_moves"):
        md("<div class='sec-title'>Biggest moves</div><div class='sec-note'>Change in Kč, positive and negative.</div>")
        top = pd.concat([per.nlargest(12, "delta"), per.nsmallest(12, "delta")])
        top = top[top.delta != 0]
        steps = list(zip(top.index, top.delta)) + [("others", per.delta.sum() - top.delta.sum())]
        fig = go.Figure(go.Waterfall(x=[s for s, _ in steps] + ["net change"], y=[v for _, v in steps] + [0],
                                     measure=["relative"] * len(steps) + ["total"],
                                     increasing=dict(marker_color=GREEN), decreasing=dict(marker_color=RED),
                                     totals=dict(marker_color=BLUE), connector=dict(line=dict(color="#ddd"))))
        fig.update_xaxes(tickangle=-60)
        st.plotly_chart(style(fig, 420), width="stretch")
        ratio = top[THIS] / top[LAST].replace(0, np.nan) - 1
        table = top.assign(change=[("new" if top[LAST].iat[k] == 0 else "no spend" if top[THIS].iat[k] == 0
                                    else f"{100 * r:+.1f} %") for k, r in enumerate(ratio)]).reset_index()
        table.columns = ["Advertiser", f"{win} {LAST} (Kč)", f"{win} {THIS} (Kč)", "Change (Kč)", "Change %"]
        st.dataframe(table, hide_index=True, width="stretch")

# =====================
# CONCENTRATION
# =====================
elif nav == "Concentration":
    with st.container(key="box_conc"):
        md("<div class='sec-title'>Concentration</div><div class='sec-note'>Share of the biggest N advertisers in the "
           "money assigned to advertisers, ranked separately for each year.</div>")
        rows = []
        for y in [LAST, THIS]:
            s = tot[y].sort_values(ascending=False)
            for k in [10, 50, 100, 200]:
                if k < (s > 0).sum():
                    rows.append({"year": y, "top": f"top {k}", "share": s.head(k).sum() / s.sum()})
        conc = pd.DataFrame(rows)
        fig = go.Figure([go.Bar(x=conc[conc.year == y].top, y=conc[conc.year == y].share, name=str(y), marker_color=c,
                                text=[f"{100 * v:.1f} %" for v in conc[conc.year == y].share], textposition="outside")
                         for y, c in [(LAST, GREY), (THIS, BLUE)]])
        fig.update_layout(barmode="group", yaxis_tickformat=".0%")
        st.plotly_chart(style(fig, 360), width="stretch")

# =====================
# OUTLOOK
# =====================
elif nav == "Outlook":
    full_last = data[data.year == LAST].groupby("advertiser").spend.sum()
    ratio_all = full_last.sum() / tot[LAST].sum()
    ratio = (full_last / tot[LAST].replace(0, np.nan)).fillna(ratio_all)
    proj = tot[THIS] * ratio.reindex(tot.index).fillna(ratio_all)
    with st.container(key="box_out"):
        md("<div class='sec-title'>Full-year outlook</div><div class='sec-note'>This year's YTD × last year's seasonality "
           "(full year / same weeks). Advertisers without last year's spend get the market's ratio, so the table does "
           "not sum to the card. Seasonal scaling, not a forecast.</div>")
        est = tot[THIS].sum() * ratio_all
        plain_card("card_proj", f"Projected {THIS}", [(f"{LAST}", czk(full_last.sum())), (f"{THIS} (est.)", czk(est))],
                   chip(est, full_last.sum()))
        out = pd.DataFrame({f"YTD {THIS}": tot[THIS], f"Projected {THIS}": proj, f"{LAST} full": full_last}
                           ).fillna(0).sort_values(f"Projected {THIS}", ascending=False).head(25)
        st.dataframe(out.round(), width="stretch")

# =====================
# SIGNALS
# =====================
elif nav == "Signals":
    def by_adv(weeks, year):
        return data[(data.year == year) & data.week.isin(weeks)].groupby("advertiser").spend.sum()

    expected = by_adv(rest, THIS) * (by_adv(w4, LAST) / by_adv(rest, LAST).replace(0, np.nan))
    drops = pd.DataFrame({"Expected 4W": expected, "Actual 4W": by_adv(w4, THIS)}).dropna()
    drops["Gap"] = drops["Actual 4W"] / drops["Expected 4W"] - 1
    drops = drops[(drops["Expected 4W"] > 20000) & (drops.Gap < -.3)].sort_values("Gap")
    with st.container(key="box_sig"):
        md(f"<div class='sec-title'>Drops — {len(drops)} advertisers</div><div class='sec-note'>Expectation = this year's "
           "YTD − 4W × the advertiser's own seasonality last year (last 4W / YTD − 4W). Flagged when the last four "
           "weeks fall more than 30 % short.</div>")
        st.dataframe(drops.round({"Expected 4W": 0, "Actual 4W": 0}).reset_index().rename(columns={"advertiser": "Advertiser"}),
                     hide_index=True, width="stretch", column_config={"Gap": st.column_config.NumberColumn(format="percent")})
