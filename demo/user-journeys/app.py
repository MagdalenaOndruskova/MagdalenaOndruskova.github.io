"""Search × Assistant: cross-service view (public demo).

An anonymised imitation of a report I build at work: how users of one product
(search) start, combine or replace it with another (an AI assistant), month by
month. Layout and styling follow the real report; the company, definitions and
numbers do NOT. Every user is simulated at start-up from a simple Markov model.

Run locally:  streamlit run app.py
In the browser it runs through stlite (see index.html).
"""
import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Search × Assistant · demo", layout="wide")

MONTHS = [f"2025-{m:02d}" for m in range(1, 13)]
LAUNCH = 6                                   # index of the month the assistant launches
STATES = ["inactive", "search only", "both", "assistant only"]
LIVE = ["search only", "both", "assistant only"]
STACK = ["assistant only", "both", "search only"]
ACCENT, RED, GREEN, ORANGE, GREY = "#1a6faf", "#c0392b", "#27ae60", "#e67e22", "#8c8c8c"
LABELS = {"search only": "search only", "both": "both (combination)", "assistant only": "assistant only",
          "inactive": "dropped out (inactive)"}
COLORS = {"search only": ACCENT, "both": RED, "assistant only": GREEN, "inactive": GREY}
SEGMENTS = ["heavy", "regular", "light"]
HW = {"all": "All (all devices)", "desktop": "Desktop", "phone": "Phone (mobile + tablet)"}

CSS = """
:root{--accent:#1a6faf;--ink:#1f2933;--muted:#667085;--line:#e3e8ee;}
.hero{background:linear-gradient(135deg,#1a6faf 0%,#15577f 100%);color:#fff;border-radius:14px;padding:22px 26px;margin:6px 0 18px;}
.hero h1{margin:0 0 6px;font-size:24px;font-weight:700;color:#fff;padding:0;}
.hero p{margin:0;opacity:.92;font-size:14px;}
.kpi-row{display:flex;flex-wrap:wrap;gap:14px;margin:10px 0 18px;}
.kpi{flex:1 1 180px;background:#fff;border:1px solid var(--line);border-radius:12px;padding:14px 16px;box-shadow:0 1px 2px rgba(16,24,40,.04);}
.kpi .k-label{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.03em;}
.kpi .k-value{font-size:26px;font-weight:700;margin:4px 0 2px;color:var(--ink);}
.kpi .k-sub{font-size:12px;color:var(--muted);}
.kpi .k-delta{font-size:12px;font-weight:600;margin-top:3px;}
.kpi .k-delta.up{color:#27ae60;} .kpi .k-delta.down{color:#c0392b;} .kpi .k-delta.flat{color:#8c8c8c;}
.kpi.good{border-left:4px solid #27ae60;} .kpi.bad{border-left:4px solid #c0392b;}
.kpi.warn{border-left:4px solid #e67e22;} .kpi.accent{border-left:4px solid var(--accent);}
.callout{border-radius:10px;padding:12px 16px;margin:12px 0;font-size:14px;line-height:1.5;color:var(--ink);}
.callout.info{background:#eaf3fb;border:1px solid #cfe2f3;}
.callout.good{background:#eafaf0;border:1px solid #c6efd6;}
.callout.warn{background:#fef6ec;border:1px solid #fad9b4;}
.callout.question{background:#eef4fb;border:1px solid #cfe0f2;border-left:4px solid var(--accent);}
.callout.summary{background:#f3f7f3;border:1px solid #d8e6d8;border-left:4px solid #27ae60;}
.callout.summary ul{margin:6px 0 0;padding-left:20px;}
.callout.howto{background:#f7f9fb;border:1px solid #e3e8ee;border-left:3px solid #99a7b5;font-size:12.5px;color:#4a5568;padding:8px 12px;}
.sec-h{font-size:19px;font-weight:700;margin:26px 0 4px;padding-top:8px;border-top:2px solid var(--line);color:var(--ink);}
.sub-h{font-size:15px;font-weight:600;color:#334155;margin:22px 0 6px;padding-top:14px;border-top:1px solid var(--line);}
.cap{font-size:12px;color:var(--muted);margin:2px 0 10px;}
"""


# =====================
# SYNTHETIC USERS
# =====================
@st.cache_data
def simulate(seed: int, n_users: int = 30000):
    rng = np.random.default_rng(seed)
    device = rng.choice(["desktop", "phone"], n_users, p=[.42, .58])
    habit = rng.gamma(2.2, 4.0, n_users)                      # typical active days per month
    curiosity = rng.beta(2, 5, n_users) * np.where(device == "phone", 1.35, 1.0)
    first = rng.choice(len(MONTHS), n_users, p=np.r_[.55, np.full(len(MONTHS) - 1, .45 / (len(MONTHS) - 1))])

    state = np.zeros((n_users, len(MONTHS)), dtype=np.int8)
    for t in range(len(MONTHS)):
        prev = state[:, t - 1] if t else np.zeros(n_users, dtype=np.int8)
        joined = first <= t
        u = rng.random(n_users)
        stay = np.clip(.75 + habit / 60, .75, .97)            # loyal users rarely drop out
        nxt = prev.copy()
        nxt[(prev == 0) & joined & (first == t)] = 1          # newcomers start in search
        nxt[(prev == 0) & joined & (first < t) & (u < .18)] = 1
        live = prev > 0
        nxt[live & (u > stay)] = 0                            # churn
        if t >= LAUNCH:
            age = t - LAUNCH + 1
            p_try = curiosity * (.10 + .05 * age)
            v = rng.random(n_users)
            nxt[(nxt == 1) & live & (v < p_try)] = 2          # start combining
            nxt[(nxt == 2) & (v < .04 + .12 * curiosity)] = 3  # assistant replaces search
            nxt[(nxt == 2) & (v > .82)] = 1                   # tried it, went back
            nxt[(nxt == 3) & (v > .78)] = 2
        state[:, t] = nxt

    days = np.clip(rng.poisson(habit[:, None] * rng.uniform(.75, 1.25, (n_users, len(MONTHS)))), 1, 30)
    days = np.where(state > 0, days, 0)
    searches = np.where(np.isin(state, [1, 2]), days * rng.gamma(3, 1.1, (n_users, 1)), 0)
    searches = np.where(state == 2, searches * .62, searches)  # some searching moves to the assistant
    chats = np.where(np.isin(state, [2, 3]), days * rng.gamma(2, .9, (n_users, 1)), 0)

    df = pd.DataFrame({
        "user": np.repeat(np.arange(n_users), len(MONTHS)),
        "month": np.tile(MONTHS, n_users),
        "device": np.repeat(device, len(MONTHS)),
        "state": np.array(STATES)[state.ravel()],
        "days": days.ravel(),
        "searches": searches.ravel().round(),
        "chats": chats.ravel().round(),
    })
    df["segment"] = np.select([df.days >= 15, df.days >= 5, df.days >= 1], SEGMENTS, "none")
    return df


# =====================
# RENDER HELPERS
# =====================
def sec(title):
    st.markdown(f"<div class='sec-h'>{html.escape(title)}</div>", unsafe_allow_html=True)


def sub(title):
    st.markdown(f"<div class='sub-h'>{html.escape(title)}</div>", unsafe_allow_html=True)


def callout(kind, text):
    st.markdown(f"<div class='callout {kind}'>{text}</div>", unsafe_allow_html=True)


def question(text):
    callout("question", f"<b>Product question:</b> {text}")


def how(text):
    callout("howto", f"<b>How it is computed:</b> {text}")


def cap(text):
    st.markdown(f"<div class='cap'>{html.escape(text)}</div>", unsafe_allow_html=True)


def kpis(items):
    cards = []
    for label, value, sub_text, kind, delta, direction in items:
        d = f"<div class='k-delta {direction}'>{html.escape(delta)}</div>" if delta else ""
        cards.append(f"<div class='kpi {kind}'><div class='k-label'>{html.escape(label)}</div>"
                     f"<div class='k-value'>{value}</div><div class='k-sub'>{html.escape(sub_text)}</div>{d}</div>")
    st.markdown(f"<div class='kpi-row'>{''.join(cards)}</div>", unsafe_allow_html=True)


def style(fig, h=380):
    fig.update_layout(template="plotly_white", height=h, margin=dict(l=60, r=20, t=40, b=46),
                      font=dict(family="-apple-system,Segoe UI,Roboto,Arial,sans-serif", size=13, color="#1f2933"),
                      legend=dict(title_text="", orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    return fig


def rgba(hex_color, alpha):
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def pct(x, dec=1):
    return f"{100 * x:.{dec}f} %"


def pp(b, a, ref):
    d = 100 * (b - a)
    return f"{d:+.1f} pp vs {ref}", "up" if d > .05 else "down" if d < -.05 else "flat"


def fmt(n):
    return f"{n:,.0f}".replace(",", " ")


# =====================
# SIDEBAR
# =====================
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


@st.dialog("About the report", width="large")
def about():
    st.markdown(
        "This is a **public demo** of the kind of cross-service report I build at work. It follows how users "
        "move between two products, **search** and an **AI assistant**: adoption, transitions between states, "
        "retention and the effect on search volume.\n\n"
        "**Everything here is simulated.** 30,000 users are generated by a small Markov model when the app "
        "starts; the product, the metric definitions and every number are made up.\n\n"
        "#### Key terms\n"
        "- **States:** *search only*, *both (combination)*, *assistant only*, *dropped out*.\n"
        "- **A vs B:** two specific months compared with each other (default: the last two).\n"
        "- **Segments:** by active days in a month: heavy 15+, regular 5–14, light 1–4. Shorter months push "
        "users into lower segments, so compare segments across months with care.")


with st.sidebar:
    if st.button("ℹ️ About the report / README", width="stretch"):
        about()
    st.header("Filters")
    hw = st.selectbox("Device", list(HW), format_func=HW.get)
    m_from, m_to = st.select_slider("Period (from – to)", MONTHS, value=(MONTHS[0], MONTHS[-1]),
                                    help="The assistant launches in the middle of the year; narrow the window here.")
    window = [m for m in MONTHS if m_from <= m <= m_to]
    month_b = st.selectbox("Main month (B)", window, index=len(window) - 1)
    month_a = st.selectbox("Comparison month (A)", window, index=max(0, len(window) - 2))
    st.markdown("**Segments (empty = all)**")
    segs = st.multiselect("Search segment in month B", SEGMENTS)
    st.markdown("**Breakdown**")
    by_segment = st.checkbox("Break charts down by segment", value=True)
    seed = st.number_input("Synthetic population", 1, 50, 3, help="Each number simulates a different population.")

data = simulate(int(seed))
if hw != "all":
    data = data[data.device == hw]
if segs:
    keep = data.loc[(data.month == month_b) & data.segment.isin(segs), "user"]
    data = data[data.user.isin(keep)]

st.markdown(
    "<div class='hero'><h1>Search × Assistant — cross-service view</h1>"
    f"<p>Main month <b>{month_b}</b> vs comparison <b>{month_a}</b> · {html.escape(HW[hw])}</p></div>",
    unsafe_allow_html=True)
callout("warn", "<b>Synthetic demo.</b> The layout follows a real report; the product, the definitions and every "
                "number are simulated.")

# =====================
# SHARED CALCULATIONS
# =====================
live = data[data.state != "inactive"]
counts = live.pivot_table(index="month", columns="state", values="user", aggfunc="count", fill_value=0)
counts = counts.reindex(index=MONTHS, columns=LIVE, fill_value=0)
shares = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)
wide = data.pivot(index="user", columns="month", values="state")


def transitions(a, b):
    return pd.crosstab(wide[a], wide[b]).reindex(index=STATES, columns=STATES, fill_value=0)


m = transitions(month_a, month_b)
adopt = shares[["both", "assistant only"]].sum(axis=1)
active_b = int(counts.loc[month_b].sum())
live_a = m.loc[LIVE].sum().sum()
churn = m.loc[LIVE, "inactive"].sum() / live_a if live_a else 0
new = m.loc["inactive", LIVE].sum() / max(m[LIVE].sum().sum(), 1)
in_asst_a = m.loc[["both", "assistant only"]].sum().sum()
left_asst = m.loc[["both", "assistant only"], ["inactive", "search only"]].sum().sum() / max(in_asst_a, 1)
ever = data[data.state.isin(["both", "assistant only"])].groupby("user").month.nunique()
one_and_done = (ever == 1).mean() if len(ever) else 0

per_user = live[live.month == month_b].groupby("state").searches.mean()
substitution = per_user.get("search only", np.nan) / per_user.get("both", np.nan)

vol = data.groupby("month").searches.sum()
search_change = vol[month_b] / vol[month_a] - 1 if vol[month_a] else 0
moved = data[(data.month.isin([month_a, month_b]))].pivot(index="user", columns="month", values="searches").fillna(0)
went = wide[(wide[month_a] == "search only") & wide[month_b].isin(["both", "assistant only"])].index
lost_to_asst = (moved.loc[moved.index.isin(went), month_a] - moved.loc[moved.index.isin(went), month_b]).sum()
lost_share = lost_to_asst / vol[month_a] if vol[month_a] else 0

asst = data.state.isin(["both", "assistant only"])
firsts = data[asst].groupby("user").month.min().rename("cohort")
coh = data[asst].join(firsts, on="user")
coh["rel"] = coh.month.map(MONTHS.index) - coh.cohort.map(MONTHS.index)
size = coh[coh.rel == 0].groupby("cohort").user.nunique()
ret = coh.groupby(["cohort", "rel"]).user.nunique().unstack().div(size, axis=0)
ret3 = ret[3].dropna().mean() if 3 in ret.columns and ret[3].notna().any() else np.nan

t_exec, t_a, t_b, t_c = st.tabs(["Executive summary", "A · Cross-service (monthly)", "B · Search volume", "C · Methodology"])

# =====================
# EXECUTIVE SUMMARY
# =====================
with t_exec:
    sec("Executive summary — simple answers")
    question("What is the overall effect of the assistant on search, and how is adoption going?")
    callout("info", "Short answers for leadership. Numbers follow the filters on the left (device, months); "
                    "the detailed breakdowns are in the other areas.")
    d_txt, d_dir = pp(adopt[month_b], adopt[month_a], month_a)
    kpis([
        ("Share of users with the assistant", pct(adopt[month_b]), f"{month_b} · {fmt(active_b)} active", "good", d_txt, d_dir),
        ("Search volume A→B", f"{100 * search_change:+.1f} %", f"{month_a} → {month_b}", "bad" if search_change < 0 else "good", "", "flat"),
        ("Searches lost to the assistant", pct(lost_share), f"of {month_a} searches, by users who moved", "warn", "", "flat"),
    ])
    kpis([
        ("Substitution vs complement", f"{substitution:.2f}×", "searches per user: search only vs both", "accent", "", "flat"),
        ("Retention in assistant M+3", pct(ret3) if ret3 == ret3 else "–", "of the first-use cohort", "warn", "", "flat"),
        ("One-and-done", pct(one_and_done), "tried the assistant in one month only", "warn", "", "flat"),
        ("Dropped out A→B", pct(churn), f"active in {month_a}, inactive in {month_b}", "accent", "", "flat"),
    ])
    trend = "grows" if adopt[month_b] > adopt[month_a] else "falls"
    callout("summary",
            "<b>What it means:</b><ul>"
            f"<li>The share of users with the assistant <b>{trend}</b> ({pct(adopt[month_a])} → {pct(adopt[month_b])} "
            f"between {month_a} and {month_b}).</li>"
            f"<li>Search volume changed by <b>{100 * search_change:+.1f} %</b>; users who moved to the assistant took "
            f"<b>{pct(lost_share)}</b> of last month's searches with them.</li>"
            f"<li><b>{pct(one_and_done)}</b> of assistant users are one-and-done: room to improve repeat value.</li></ul>")
    how("states come from each user's activity in the month; searches lost to the assistant = the drop in searches "
        "of users who went from <i>search only</i> to the assistant, as a share of all searches in month A.")

# =====================
# A · CROSS-SERVICE
# =====================
with t_a:
    s1, s2, s3, s4 = st.tabs(["Summary", "Share with assistant", "Transitions", "Retention"])
    with s1:
        sec("1. Summary for stakeholders")
        question("How do users migrate between our services?")
        kpis([
            ("Active users", fmt(active_b), f"unique across search + assistant · {month_b}", "accent", "", "flat"),
            ("Share with assistant", pct(adopt[month_b]), "", "good", *pp(adopt[month_b], adopt[month_a], month_a)),
            ("Dropped out A→B", pct(churn), f"{month_a}→{month_b}", "bad", "", "flat"),
            ("New A→B", pct(new), f"unseen in {month_a}, active in {month_b}", "good", "", "flat"),
            ("Left the assistant A→B", pct(left_asst), f"in assistant in {month_a}, not in {month_b}", "warn", "", "flat"),
        ])
        row = []
        for s, kind in [("search only", "accent"), ("both", "bad"), ("assistant only", "good")]:
            row.append((LABELS[s], fmt(counts.loc[month_b, s]),
                        f"share {pct(shares.loc[month_b, s])} · {month_a}: {fmt(counts.loc[month_a, s])}",
                        kind, *pp(shares.loc[month_b, s], shares.loc[month_a, s], month_a)))
        kpis(row)
        callout("good", f"The share of users who use the assistant moves from {pct(adopt[month_a])} to "
                        f"{pct(adopt[month_b])}. Of those in the assistant in {month_a}, {pct(left_asst)} are not there in {month_b}.")

    with s2:
        sec("2. Share of users with the assistant")
        question("Which users adopt the assistant, and how fast?")
        fig = go.Figure()
        for s in STACK:
            fig.add_scatter(x=window, y=shares.loc[window, s], name=LABELS[s], stackgroup="one",
                            line=dict(color=COLORS[s], width=1), fillcolor=rgba(COLORS[s], .55))
        fig.update_layout(yaxis_tickformat=".0%", xaxis_type="category")
        st.plotly_chart(style(fig), width="stretch")
        cap("Share of active users by how they use the two products.")
        if by_segment:
            sub("By search segment")
            by_seg = live[live.month.isin(window) & live.segment.isin(SEGMENTS)]
            seg_share = (by_seg.assign(a=by_seg.state.isin(["both", "assistant only"]))
                         .groupby(["segment", "month"]).a.mean().unstack(0).reindex(columns=SEGMENTS))
            fig = go.Figure([go.Scatter(x=seg_share.index, y=seg_share[s], name=s, mode="lines+markers",
                                        line=dict(color=c, width=2.5))
                             for s, c in zip(SEGMENTS, [ACCENT, ORANGE, GREY])])
            fig.update_layout(yaxis_tickformat=".0%", xaxis_type="category")
            st.plotly_chart(style(fig, 320), width="stretch")
            cap("Share of users with the assistant within each segment. Heavier users adopt sooner.")

    with s3:
        sec("3. Transitions between states")
        question(f"Where do users from each state in {month_a} end up in {month_b}?")
        src, dst, val, col = [], [], [], []
        for i, a in enumerate(STATES):
            for j, b in enumerate(STATES):
                v = int(m.loc[a, b])
                if v and not (a == b == "inactive"):
                    src.append(i); dst.append(len(STATES) + j); val.append(v); col.append(rgba(COLORS[a], .35))
        fig = go.Figure(go.Sankey(
            node=dict(label=[f"{LABELS[s]} ({month_a})" for s in STATES] + [f"{LABELS[s]} ({month_b})" for s in STATES],
                      color=[COLORS[s] for s in STATES] * 2, pad=18, thickness=14),
            link=dict(source=src, target=dst, value=val, color=col),
            textfont=dict(color="#1f2933", size=13, shadow="none")))
        st.plotly_chart(style(fig, 440), width="stretch")
        norm = m.div(m.sum(axis=1).replace(0, np.nan), axis=0)
        fig = go.Figure(go.Heatmap(z=norm.values, x=[f"→ {LABELS[s]}" for s in STATES], y=[LABELS[s] for s in STATES],
                                   colorscale=[[0, "#f6f8fb"], [1, ACCENT]], zmin=0, zmax=1, showscale=False,
                                   text=norm.map(lambda x: "" if pd.isna(x) else f"{100 * x:.0f} %").values,
                                   texttemplate="%{text}"))
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(style(fig, 300), width="stretch")
        how("rows are users by their state in month A, columns their state in month B; each row sums to 100 %.")

    with s4:
        sec("4. Retention in the assistant")
        question("Do people who try the assistant come back?")
        r = ret.loc[:, ret.columns <= 5]
        fig = go.Figure(go.Heatmap(z=r.values, x=[f"M+{c}" for c in r.columns], y=list(r.index),
                                   colorscale=[[0, "#f6f8fb"], [1, GREEN]], zmin=0, zmax=1, showscale=False,
                                   text=r.map(lambda x: "" if pd.isna(x) else f"{100 * x:.0f} %").values,
                                   texttemplate="%{text}"))
        fig.update_yaxes(autorange="reversed", type="category")
        st.plotly_chart(style(fig, 340), width="stretch")
        how("cohort = month of first assistant use; cells show the share of the cohort active in the assistant "
            "n months later. Recent cohorts are censored at the edge of the window.")

# =====================
# B · SEARCH VOLUME
# =====================
with t_b:
    sec("Search volume A → B, decomposed")
    question("How much search does the assistant take away?")
    a = data[data.month == month_a].set_index("user")
    b = data[data.month == month_b].set_index("user")
    j = a[["state", "searches"]].join(b[["state", "searches"]], lsuffix="_a", rsuffix="_b", how="outer").fillna(
        {"searches_a": 0, "searches_b": 0, "state_a": "inactive", "state_b": "inactive"})
    d = j.searches_b - j.searches_a
    steps = [
        ("New or returning", d[(j.state_a == "inactive") & (j.state_b != "inactive")].sum()),
        ("Dropped out", d[(j.state_a != "inactive") & (j.state_b == "inactive")].sum()),
        ("Moved to assistant", d[(j.state_a == "search only") & j.state_b.isin(["both", "assistant only"])].sum()),
        ("Back to search", d[j.state_a.isin(["both", "assistant only"]) & (j.state_b == "search only")].sum()),
    ]
    steps.append(("Same behaviour", d.sum() - sum(v for _, v in steps)))
    fig = go.Figure(go.Waterfall(
        x=[f"Searches {month_a}"] + [s for s, _ in steps] + [f"Searches {month_b}"],
        measure=["absolute"] + ["relative"] * len(steps) + ["total"],
        y=[j.searches_a.sum()] + [v for _, v in steps] + [0],
        increasing=dict(marker_color=GREEN), decreasing=dict(marker_color=RED),
        totals=dict(marker_color=ACCENT), connector=dict(line=dict(color="#c7ccd1"))))
    st.plotly_chart(style(fig, 400), width="stretch")
    how("each user's change in searches between A and B is assigned to one group by their states in A and B; "
        "the groups add up exactly to the total change.")

# =====================
# C · METHODOLOGY
# =====================
with t_c:
    sec("Methodology and glossary")
    st.markdown(
        "- **Active user** – at least one day with search or the assistant in the month.\n"
        "- **States** – *search only*, *both*, *assistant only*; *dropped out* = active in A, not in B.\n"
        "- **Share with assistant** – (both + assistant only) / active users.\n"
        "- **Substitution vs complement** – searches per user in *search only* divided by *both*. "
        "Above 1 means users who combine search less: the assistant substitutes part of search.\n"
        "- **One-and-done** – assistant users active in it in exactly one month of the window.\n"
        "- **Retention M+n** – share of a first-use cohort still in the assistant n months later.")
    callout("warn", "<b>Simulated data.</b> Users, behaviour and every number on this page are generated; "
                    "only the questions and the structure come from real work.")
