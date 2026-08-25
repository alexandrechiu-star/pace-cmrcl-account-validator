import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import load_accounts
from src.enricher import enrich_sample
# Scoring runs via a Claude Code session for now (pending sanctioned API access),
# not a live call from this app — see Tab 3. src/tier_scorer.py is unchanged and
# ready to wire back in once that access is approved.

st.set_page_config(
    page_title="PACE CMRCL Account Signal Validator",
    page_icon="☁️",
    layout="wide"
)

SF_BLUE = "#0070D2"
SF_NAVY = "#032D60"
SF_ORANGE = "#FFB75D"
SF_GREEN = "#4BC076"
SF_RED = "#E96D76"

st.markdown(f"""
<style>
    .main-header {{
        color: {SF_NAVY};
        font-size: 28px;
        font-weight: bold;
        margin-bottom: 0px;
    }}
    .sub-header {{
        color: {SF_BLUE};
        font-size: 16px;
        margin-bottom: 20px;
    }}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load whichever data is available"""
    if os.path.exists('data/scored_accounts.csv'):
        return pd.read_csv('data/scored_accounts.csv')
    elif os.path.exists('data/enriched_accounts.csv'):
        return pd.read_csv('data/enriched_accounts.csv')
    else:
        return load_accounts()


def main():
    st.markdown('<p class="main-header">PACE CMRCL Account Signal Validator</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">External signal enrichment to detect tier mismatches and surface account opportunities</p>', unsafe_allow_html=True)

    df = load_data()
    has_ai_scores = 'mismatch_category' in df.columns

    st.sidebar.header("Filters")

    avp_options = ['All'] + sorted(df['AVP_REGION'].dropna().unique().tolist())
    selected_avp = st.sidebar.selectbox("AVP Region", avp_options)

    tier_options = sorted(df['ACCT_TIER'].dropna().unique().tolist())
    selected_tier = st.sidebar.multiselect("Current Tier", tier_options, default=tier_options)

    selected_mismatch = 'All'
    if has_ai_scores:
        mismatch_options = ['All', 'Overtiered Risk', 'Undertiered Opportunity', 'Bad Data', 'Tier Confirmed']
        selected_mismatch = st.sidebar.selectbox("Mismatch Category", mismatch_options)

    filtered = df.copy()
    if selected_avp != 'All':
        filtered = filtered[filtered['AVP_REGION'] == selected_avp]
    if selected_tier:
        filtered = filtered[filtered['ACCT_TIER'].isin(selected_tier)]
    if has_ai_scores and selected_mismatch != 'All':
        filtered = filtered[filtered['mismatch_category'] == selected_mismatch]

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Tier Distribution",
        "🔍 Tier vs. Performance",
        "🎯 Mismatch Detection",
        "📋 Account Detail"
    ])

    with tab1:
        st.subheader("Current Account Tier Distribution")

        col1, col2, col3, col4, col5 = st.columns(5)
        tier_counts = filtered['ACCT_TIER'].value_counts()

        for col, tier in zip([col1, col2, col3, col4, col5],
                              ['Tier 1', 'Tier 2', 'Tier 3', 'Tier 4', 'Untiered']):
            with col:
                count = tier_counts.get(tier, 0)
                pct = count / len(filtered) * 100 if len(filtered) else 0
                st.metric(tier, f"{count:,}", f"{pct:.1f}% of accounts")

        col_left, col_right = st.columns(2)

        with col_left:
            fig_pie = px.pie(
                filtered,
                names='ACCT_TIER',
                title='Account Tier Distribution',
                color_discrete_sequence=[SF_BLUE, SF_ORANGE, SF_GREEN, SF_RED, '#CCCCCC']
            )
            fig_pie.update_layout(title_font_color=SF_NAVY)
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_right:
            aov_by_tier = filtered.groupby('ACCT_TIER')['FY27_C360_AOV'].sum().reset_index()
            aov_by_tier.columns = ['Tier', 'Total AOV']
            aov_by_tier = aov_by_tier.sort_values('Tier')

            fig_aov = px.bar(
                aov_by_tier,
                x='Tier',
                y='Total AOV',
                title='Total AOV by Tier ($)',
                color='Tier',
                color_discrete_sequence=[SF_BLUE, SF_ORANGE, SF_GREEN, SF_RED]
            )
            fig_aov.update_layout(title_font_color=SF_NAVY, showlegend=False)
            st.plotly_chart(fig_aov, use_container_width=True)

        st.subheader("Tier Distribution by AVP Region")
        tier_region = pd.crosstab(
            filtered['AVP_REGION'],
            filtered['ACCT_TIER'],
            normalize='index'
        ) * 100

        fig_heat = px.imshow(
            tier_region.round(1),
            text_auto=True,
            color_continuous_scale='Blues',
            title='% of Accounts by Tier within Each AVP Region'
        )
        fig_heat.update_layout(title_font_color=SF_NAVY)
        st.plotly_chart(fig_heat, use_container_width=True)

    with tab2:
        st.subheader("Does Current Tier Predict Revenue?")
        st.caption("The core question: are higher-tier accounts actually generating more ACV?")

        col_left, col_right = st.columns(2)

        with col_left:
            acv_by_tier = filtered.groupby('ACCT_TIER').agg(
                avg_acv=('H1_FY27_ACV', 'mean'),
                median_acv=('H1_FY27_ACV', 'median'),
                pct_with_acv=('H1_FY27_ACV', lambda x: (x > 0).mean() * 100),
                count=('H1_FY27_ACV', 'count')
            ).reset_index().sort_values('ACCT_TIER')

            fig_acv = px.bar(
                acv_by_tier,
                x='ACCT_TIER',
                y='avg_acv',
                title='Average H1 FY27 ACV by Tier ($)',
                color='ACCT_TIER',
                color_discrete_sequence=[SF_BLUE, SF_ORANGE, SF_GREEN, SF_RED],
                text='avg_acv'
            )
            fig_acv.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
            fig_acv.update_layout(title_font_color=SF_NAVY, showlegend=False)
            st.plotly_chart(fig_acv, use_container_width=True)

        with col_right:
            fig_pct = px.bar(
                acv_by_tier,
                x='ACCT_TIER',
                y='pct_with_acv',
                title='% of Accounts That Closed Any ACV in H1 FY27',
                color='ACCT_TIER',
                color_discrete_sequence=[SF_BLUE, SF_ORANGE, SF_GREEN, SF_RED],
                text='pct_with_acv'
            )
            fig_pct.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_pct.update_layout(title_font_color=SF_NAVY, showlegend=False, yaxis_title='% with ACV')
            st.plotly_chart(fig_pct, use_container_width=True)

        st.subheader("⚠️ Potential Undertiering — High ACV, Low Tier")
        st.caption("Accounts generating significant ACV despite being Tier 3/4 — candidates for upgrade")

        undertiered = filtered[
            (filtered['ACCT_TIER'].isin(['Tier 3', 'Tier 4', 'Untiered'])) &
            (filtered['H1_FY27_ACV'] > 50000)
        ].sort_values('H1_FY27_ACV', ascending=False)[
            ['COMBO_COMPANY_NAME', 'ACCT_TIER', 'H1_FY27_ACV',
             'FY27_C360_AOV', 'ATR', 'COMBO_LOCKED_INDUSTRY',
             'ACCT_OWN_FULL_NM', 'AVP_REGION']
        ].head(20)

        st.dataframe(
            undertiered,
            use_container_width=True,
            hide_index=True,
            column_config={
                'H1_FY27_ACV': st.column_config.NumberColumn('H1 FY27 ACV', format='$%,.0f'),
                'FY27_C360_AOV': st.column_config.NumberColumn('C360 AOV', format='$%,.0f'),
                'ATR': st.column_config.NumberColumn('ATR', format='$%,.0f'),
            }
        )

        st.subheader("⚠️ Potential Overtiering — No ACV, High Tier")
        st.caption("Tier 1/2 accounts with zero H1 FY27 ACV — candidates for review")

        overtiered = filtered[
            (filtered['ACCT_TIER'].isin(['Tier 1', 'Tier 2'])) &
            (filtered['H1_FY27_ACV'] == 0)
        ].sort_values('FY27_C360_AOV', ascending=False)[
            ['COMBO_COMPANY_NAME', 'ACCT_TIER', 'H1_FY27_ACV',
             'FY27_C360_AOV', 'ATR', 'COMBO_LOCKED_INDUSTRY',
             'ACCT_OWN_FULL_NM', 'AVP_REGION']
        ].head(20)

        st.dataframe(
            overtiered,
            use_container_width=True,
            hide_index=True,
            column_config={
                'H1_FY27_ACV': st.column_config.NumberColumn('H1 FY27 ACV', format='$%,.0f'),
                'FY27_C360_AOV': st.column_config.NumberColumn('C360 AOV', format='$%,.0f'),
                'ATR': st.column_config.NumberColumn('ATR', format='$%,.0f'),
            }
        )

    with tab3:
        if os.path.exists('data/enriched_accounts.csv') and not has_ai_scores:
            st.info(
                "Accounts are enriched but not yet analyzed. Mismatch detection runs "
                "through a Claude Code session rather than a live API call. "
                "Ask Claude Code to analyze `data/enriched_accounts.csv` "
                "and write `data/scored_accounts.csv`, then refresh this page."
            )
            if st.button("🔄 Refresh (checks for scored_accounts.csv)"):
                st.cache_data.clear()
                st.rerun()
        elif not has_ai_scores:
            st.info(
                "Step 1 of 2: enrich a sample of accounts with external signals (NewsAPI). "
                "Mismatch detection happens separately in a Claude Code session — see below."
            )

            col1, col2 = st.columns(2)
            with col1:
                sample_size = st.slider("Sample size", 50, 500, 100)
            with col2:
                st.write("")
                st.write("")
                if st.button("📡 Run Enrichment", type="primary"):
                    with st.spinner("Enriching accounts with external signals..."):
                        accounts_df = load_accounts()
                        enrich_sample(accounts_df, sample_size)
                    st.success("✅ Enrichment complete. Refreshing...")
                    st.cache_data.clear()
                    st.rerun()

            st.caption(
                "Step 2 of 2 (manual): once enriched, ask Claude Code to detect mismatches on "
                "`data/enriched_accounts.csv` and write `data/scored_accounts.csv`."
            )
        else:
            scored_df = filtered[filtered['ai_scored'] == True].copy()

            st.subheader(f"Tier Mismatch Detection — {len(scored_df)} accounts analyzed")

            col1, col2, col3, col4 = st.columns(4)

            overtiered = scored_df[scored_df['mismatch_category'] == 'Overtiered Risk']
            undertiered = scored_df[scored_df['mismatch_category'] == 'Undertiered Opportunity']
            bad_data = scored_df[scored_df['mismatch_category'] == 'Bad Data']
            confirmed = scored_df[scored_df['mismatch_category'] == 'Tier Confirmed']

            n = max(len(scored_df), 1)
            with col1:
                st.metric("🔴 Overtiered Risk", len(overtiered), f"{len(overtiered)/n*100:.1f}% of analyzed")
            with col2:
                st.metric("🟢 Undertiered Oppty", len(undertiered), f"{len(undertiered)/n*100:.1f}% of analyzed")
            with col3:
                st.metric("⚠️ Bad Data", len(bad_data), f"{len(bad_data)/n*100:.1f}% of analyzed")
            with col4:
                st.metric("✅ Tier Confirmed", len(confirmed), f"{len(confirmed)/n*100:.1f}% of analyzed")

            st.subheader("Mismatch Distribution by Current Tier")

            mismatch_by_tier = pd.crosstab(
                scored_df['ACCT_TIER'],
                scored_df['mismatch_category'],
                normalize='index'
            ) * 100

            fig_mismatch = px.imshow(
                mismatch_by_tier.round(1),
                text_auto=True,
                color_continuous_scale='RdYlGn_r',
                title='% of Accounts by Mismatch Category within Each Tier'
            )
            fig_mismatch.update_layout(title_font_color=SF_NAVY)
            st.plotly_chart(fig_mismatch, use_container_width=True)

            st.subheader("🟢 Undertiered Opportunities")
            st.caption("Low-tier accounts with strong external signals — candidates for upgrade consideration")

            undertiered_display = undertiered.sort_values('confidence', ascending=False)[
                ['COMBO_COMPANY_NAME', 'ACCT_TIER', 'confidence',
                 'H1_FY27_ACV', 'COMBO_EMPLOYEE_COUNT', 'key_signals',
                 'reasoning', 'action_recommended', 'AVP_REGION']
            ].head(25)

            st.dataframe(
                undertiered_display,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'ACCT_TIER': st.column_config.TextColumn('Current Tier'),
                    'confidence': st.column_config.TextColumn('Confidence'),
                    'H1_FY27_ACV': st.column_config.NumberColumn('H1 ACV', format='$%,.0f'),
                    'COMBO_EMPLOYEE_COUNT': st.column_config.NumberColumn('Employees', format='%,.0f'),
                    'key_signals': st.column_config.TextColumn('Key Signals', width='large'),
                    'reasoning': st.column_config.TextColumn('Reasoning', width='large'),
                    'action_recommended': st.column_config.TextColumn('Action', width='medium'),
                }
            )

            st.subheader("🔴 Overtiered Risk")
            st.caption("High-tier accounts with weak external signals — candidates for downgrade review")

            overtiered_display = overtiered.sort_values('confidence', ascending=False)[
                ['COMBO_COMPANY_NAME', 'ACCT_TIER', 'confidence',
                 'H1_FY27_ACV', 'FY27_C360_AOV', 'key_signals',
                 'reasoning', 'action_recommended', 'AVP_REGION']
            ].head(25)

            st.dataframe(
                overtiered_display,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'ACCT_TIER': st.column_config.TextColumn('Current Tier'),
                    'confidence': st.column_config.TextColumn('Confidence'),
                    'H1_FY27_ACV': st.column_config.NumberColumn('H1 ACV', format='$%,.0f'),
                    'FY27_C360_AOV': st.column_config.NumberColumn('AOV', format='$%,.0f'),
                    'key_signals': st.column_config.TextColumn('Key Signals', width='large'),
                    'reasoning': st.column_config.TextColumn('Reasoning', width='large'),
                }
            )

            st.subheader("⚠️ Bad Data Cleanup")
            st.caption("Accounts flagged for removal or consolidation")

            bad_data_display = bad_data[
                ['COMBO_COMPANY_NAME', 'ACCT_TIER', 'key_signals',
                 'reasoning', 'action_recommended', 'AVP_REGION']
            ].head(25)

            st.dataframe(
                bad_data_display,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'key_signals': st.column_config.TextColumn('Flags', width='large'),
                    'reasoning': st.column_config.TextColumn('Why Bad Data', width='large'),
                    'action_recommended': st.column_config.TextColumn('Action', width='medium'),
                }
            )

    with tab4:
        st.subheader("Account Detail Search")

        search = st.text_input("Search by company name", placeholder="e.g. Luminator, ChargePoint...")

        if search:
            results = filtered[
                filtered['COMBO_COMPANY_NAME'].str.contains(search, case=False, na=False)
            ]

            if len(results) == 0:
                st.warning(f"No accounts found matching '{search}'")
            else:
                for _, account in results.iterrows():
                    with st.expander(f"**{account['COMBO_COMPANY_NAME']}** — {account['ACCT_TIER']}"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Current Tier", account['ACCT_TIER'])
                            st.metric("Industry", account.get('COMBO_LOCKED_INDUSTRY', 'Unknown'))
                            st.metric("Location", f"{account.get('COMBO_LOCKED_CITY','')}, {account.get('COMBO_LOCKED_STATE','')}")
                        with col2:
                            st.metric("H1 FY27 ACV", f"${account.get('H1_FY27_ACV', 0):,.0f}")
                            st.metric("C360 AOV", f"${account.get('FY27_C360_AOV', 0):,.0f}")
                            st.metric("ATR", f"${account.get('ATR', 0):,.0f}")
                        with col3:
                            st.metric("AE Owner", account.get('ACCT_OWN_FULL_NM', 'Unknown'))
                            st.metric("AVP Region", account.get('AVP_REGION', 'Unknown'))
                            emp = account.get('COMBO_EMPLOYEE_COUNT')
                            st.metric("Employees", f"{emp:,.0f}" if pd.notna(emp) else 'Unknown')

                        if has_ai_scores and pd.notna(account.get('mismatch_category')):
                            st.divider()
                            st.subheader("🤖 Mismatch Analysis")
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("Mismatch Category", account.get('mismatch_category', 'N/A'))
                                st.metric("Confidence", account.get('confidence', 'N/A'))
                                st.metric("Action Recommended", account.get('action_recommended', 'N/A'))
                            with col_b:
                                st.write("**Key Signals:**")
                                signals = str(account.get('key_signals', '')).split(' | ')
                                for signal in signals:
                                    if signal:
                                        st.write(f"• {signal}")

                            if account.get('reasoning'):
                                st.info(f"**Analysis:** {account['reasoning']}")

                            if account.get('news_headlines'):
                                st.write("**Recent News:**")
                                headlines = str(account['news_headlines']).split(' | ')
                                for h in headlines:
                                    if h:
                                        st.write(f"📰 {h}")

        st.divider()
        st.subheader("Export")

        export_df = filtered.copy()
        csv = export_df.to_csv(index=False)
        st.download_button(
            "⬇️ Download Filtered Data as CSV",
            csv,
            "account_tiering_analysis.csv",
            "text/csv"
        )


if __name__ == "__main__":
    main()
