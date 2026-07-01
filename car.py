import streamlit as st
import pandas as pd
import os
import numpy as np

# 设置页面为宽屏模式
st.set_page_config(page_title="运营中心数据看板", layout="wide")


# ===========================
# 1. 数据处理与清洗
# ===========================
@st.cache_data
def load_and_process_data(file_path):
    column_names = [
        '日期', '星期', '毛利目标', '单均毛利', '发单量', '完单量', '订单GMV', '毛利率', '特惠占比',
        '订单平台抽成', '渠道活动补贴', '免佣卡收益', '第三方抽佣', '高德抽佣', '独补', '独补补贴率',
        '共补', '共补补贴率', 'C补', 'C补补贴率', '平台免佣GMV', '平台免佣GMV占比',
        '免佣订单占比', '减免GMV', '减免GMV占比', 'SP免佣成本', 'SP免佣补贴率'
    ]

    sheets = pd.read_excel(file_path, sheet_name=None, header=None)
    all_dfs = []

    for sheet_name, df in sheets.items():
        city_name = sheet_name.replace('6月', '').replace('（总毛利）', '')
        df = df.iloc[3:].copy()
        if df.shape[1] > len(column_names):
            df = df.iloc[:, :len(column_names)]
        df.columns = column_names
        df['城市'] = city_name
        df['日期'] = pd.to_datetime(df['日期'], errors='coerce')
        df = df.dropna(subset=['日期'])
        all_dfs.append(df)

    full_df = pd.concat(all_dfs, ignore_index=True)
    numeric_cols = full_df.columns.drop(['日期', '星期', '城市'])
    for col in numeric_cols:
        full_df[col] = pd.to_numeric(full_df[col], errors='coerce')

    full_df = full_df.sort_values(by=['城市', '日期'])
    full_df['计算后总毛利'] = full_df['订单GMV'] * full_df['毛利率']
    full_df['完单率'] = full_df['完单量'] / full_df['发单量']

    full_df['特惠加权分子'] = full_df['特惠占比'] * full_df['完单量']
    full_df['免佣订单加权分子'] = full_df['免佣订单占比'] * full_df['完单量']

    return full_df


# ===========================
# 2. 时间维度聚合工具函数（修复了月折线图）
# ===========================
def aggregate_by_time(df, time_dim):
    sum_cols = [
        '发单量', '完单量', '订单GMV', '订单平台抽成', '渠道活动补贴',
        '免佣卡收益', '第三方抽佣', '高德抽佣', '独补', '共补', 'C补',
        '平台免佣GMV', '减免GMV', 'SP免佣成本', '计算后总毛利',
        '特惠加权分子', '免佣订单加权分子'
    ]

    if time_dim == "日 (最近单日)":
        daily = df.groupby('日期').agg({**{k: 'sum' for k in sum_cols}}).reset_index()
        daily = daily.dropna(subset=['完单量'])
        daily['特惠占比'] = daily['特惠加权分子'] / daily['完单量']
        daily['免佣订单占比'] = daily['免佣订单加权分子'] / daily['完单量']
        daily['完单率'] = daily['完单量'] / daily['发单量']
        daily['单均毛利'] = daily['计算后总毛利'] / daily['完单量']
        daily['毛利率'] = daily['计算后总毛利'] / daily['订单GMV']
        daily['平台免佣GMV占比'] = daily['平台免佣GMV'] / daily['订单GMV']
        daily['减免GMV占比'] = daily['减免GMV'] / daily['订单GMV']

        daily_asc = daily.sort_values('日期')
        daily_asc['完单量日环比'] = daily_asc['完单量'].pct_change() * 100
        daily = daily_asc.sort_values('日期', ascending=False)

        latest_data = daily.iloc[0]
        trend_data = daily.sort_values('日期').tail(15)
        delta_label = "日环比"
        return latest_data, trend_data, delta_label

    elif time_dim == "周 (最近自然周)":
        weekly = df.groupby(pd.Grouper(key='日期', freq='W-MON')).agg({**{k: 'sum' for k in sum_cols}}).reset_index()
        weekly = weekly.dropna(subset=['完单量'])
        if weekly.empty:
            return None, None, None

        weekly['特惠占比'] = weekly['特惠加权分子'] / weekly['完单量']
        weekly['免佣订单占比'] = weekly['免佣订单加权分子'] / weekly['完单量']
        weekly['完单率'] = weekly['完单量'] / weekly['发单量']
        weekly['单均毛利'] = weekly['计算后总毛利'] / weekly['完单量']
        weekly['毛利率'] = weekly['计算后总毛利'] / weekly['订单GMV']
        weekly['平台免佣GMV占比'] = weekly['平台免佣GMV'] / weekly['订单GMV']
        weekly['减免GMV占比'] = weekly['减免GMV'] / weekly['订单GMV']

        weekly_asc = weekly.sort_values('日期')
        latest_week_start = weekly_asc['日期'].iloc[-1]
        last_week_data = df[(df['日期'] >= latest_week_start) & (df['日期'] < latest_week_start + pd.Timedelta(days=7))]
        days_count = last_week_data['日期'].nunique()

        weekly_asc['完单量周环比'] = weekly_asc['完单量'].pct_change() * 100
        if days_count < 7:
            weekly_asc.loc[weekly_asc['日期'] == latest_week_start, '完单量周环比'] = np.nan

        weekly = weekly_asc.sort_values('日期', ascending=False)

        latest_data = weekly.iloc[0]
        trend_data = weekly.sort_values('日期').tail(4)
        delta_label = "周环比"
        return latest_data, trend_data, delta_label

    elif time_dim == "月 (整月累计)":
        monthly = df.groupby(pd.Grouper(key='日期', freq='M')).agg({**{k: 'sum' for k in sum_cols}}).reset_index()
        monthly = monthly.dropna(subset=['完单量'])
        if monthly.empty:
            return None, None, None

        monthly['特惠占比'] = monthly['特惠加权分子'] / monthly['完单量']
        monthly['免佣订单占比'] = monthly['免佣订单加权分子'] / monthly['完单量']
        monthly['完单率'] = monthly['完单量'] / monthly['发单量']
        monthly['单均毛利'] = monthly['计算后总毛利'] / monthly['完单量']
        monthly['毛利率'] = monthly['计算后总毛利'] / monthly['订单GMV']
        monthly['平台免佣GMV占比'] = monthly['平台免佣GMV'] / monthly['订单GMV']
        monthly['减免GMV占比'] = monthly['减免GMV'] / monthly['订单GMV']

        monthly_asc = monthly.sort_values('日期')
        monthly_asc['完单量月环比'] = np.nan
        monthly = monthly_asc.sort_values('日期', ascending=False)

        latest_data = monthly.iloc[0]

        # ⚠️ 核心修复：这里的 trend_data 必须按天聚合后取尾数，不能直接用未聚合的 df！
        daily = df.groupby('日期').agg({**{k: 'sum' for k in sum_cols}}).reset_index()
        daily = daily.dropna(subset=['完单量'])
        trend_data = daily.sort_values('日期').tail(30)

        delta_label = "月度累计 (本月汇总)"
        return latest_data, trend_data, delta_label


# ===========================
# 3. 看板主逻辑
# ===========================
def main():
    data_path = os.path.join(os.path.dirname(__file__), 'data', '毛利目标.xlsx')
    if not os.path.exists(data_path):
        st.error(f"❌ 找不到数据文件，请确认文件位于：`{data_path}`")
        st.stop()

    try:
        df = load_and_process_data(data_path)
    except Exception as e:
        st.error(f"❌ 读取数据时发生错误：{e}")
        st.stop()

    cities = df['城市'].unique()
    st.sidebar.title("🚦 运营中心")

    city_options = ["全部城市 (总体看板)"] + list(cities)
    selected_city = st.sidebar.selectbox("🌆 选择分析范围", city_options)
    time_dim = st.sidebar.radio("⏱️ 时间维度", ["日 (最近单日)", "周 (最近自然周)", "月 (整月累计)"])

    if selected_city == "全部城市 (总体看板)":
        df_filtered = df.copy()
        title_prefix = "多城市运营分析看板"
        sub_prefix = "覆盖全部 7 个城市的运营聚合分析"
    else:
        df_filtered = df[df['城市'] == selected_city].copy()
        title_prefix = f"{selected_city} 运营分析"
        sub_prefix = f"{selected_city} 单城深度运营分析"

    latest_data, trend_data, delta_label = aggregate_by_time(df_filtered, time_dim)
    if latest_data is None:
        st.stop()

    if time_dim == "日 (最近单日)":
        date_label = latest_data['日期'].strftime('%Y年%m月%d日')
    elif time_dim == "周 (最近自然周)":
        week_start = latest_data['日期']
        week_end = week_start + pd.Timedelta(days=6)
        max_date = df_filtered['日期'].max().strftime('%m-%d')
        date_label = f"最近一周 ({week_start.strftime('%m-%d')}~{week_end.strftime('%m-%d')}, 截止至 {max_date})"
    else:
        date_label = f"2026年6月整月累计"

    st.title(f"🚀 {title_prefix}")
    st.caption(f"{sub_prefix} | {date_label} 快照")

    col1, col2, col3, col4, col5 = st.columns(5)

    if time_dim == "日 (最近单日)":
        c_metric = latest_data.get('完单量日环比', np.nan)
    elif time_dim == "周 (最近自然周)":
        c_metric = latest_data.get('完单量周环比', np.nan)
    else:
        c_metric = latest_data.get('完单量月环比', np.nan)

    if pd.isna(c_metric):
        if time_dim == "周 (最近自然周)":
            delta_str = "本周数据不完整"
        elif time_dim == "月 (整月累计)":
            delta_str = "当月仅单月数据"
        else:
            delta_str = "-"
    else:
        delta_str = f"{c_metric:.2f}% {delta_label}"

    gmv = latest_data['订单GMV']
    gmv_format = f"¥{gmv / 10000:.2f}万" if gmv > 10000 else f"¥{gmv:.0f}"

    with col1:
        st.metric(label="完单量", value=f"{int(latest_data['完单量']):,}", delta=delta_str)
    with col2:
        st.metric(label="订单 GMV", value=gmv_format, delta=delta_str)
    with col3:
        st.metric(label="毛利率", value=f"{latest_data['毛利率'] * 100:.2f}%",
                  delta=f"特惠 {latest_data['特惠占比'] * 100:.1f}%")
    with col4:
        st.metric(label="单均毛利", value=f"¥{latest_data['单均毛利']:.2f}")
    with col5:
        st.markdown("""
        <div style="background-color: #E8F0FE; padding: 10px; border-radius: 10px; text-align: center;">
            <p style="font-size: 14px; color: #555; margin: 0;">完单率</p>
            <p style="font-size: 28px; font-weight: bold; margin: 0;">{:.1%}</p>
        </div>
        """.format(latest_data['完单率']), unsafe_allow_html=True)

    st.markdown("---")

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.metric("发单量", f"{int(latest_data['发单量']):,}")
    with col_b:
        st.metric("平台免佣GMV占比", f"{latest_data['平台免佣GMV占比'] * 100:.2f}%")
    with col_c:
        st.metric("SP免佣成本", f"¥{latest_data['SP免佣成本'] / 10000:.2f}万")
    with col_d:
        st.metric("减免GMV占比", f"{latest_data['减免GMV占比'] * 100:.2f}%")

    st.markdown("---")

    l_col, r_col = st.columns([2, 1])
    with l_col:
        st.subheader("📈 营收规模趋势分析")
        st.line_chart(trend_data, x='日期', y='订单GMV', color='#FF4B4B')

        if time_dim == "月 (整月累计)":
            st.caption("趋势图展示的是6月份每日的营收走势")
        elif time_dim == "周 (最近自然周)":
            st.caption("趋势图展示的是最近4个自然周的营收走势")

    with r_col:
        st.subheader("📊 平台抽成与收益分析")
        gmv_curr = latest_data['订单GMV'] if latest_data['订单GMV'] != 0 else 1

        st.write("**订单平台抽成**")
        pct = min(1.0, latest_data['订单平台抽成'] / gmv_curr)
        st.progress(pct)
        st.caption(f"¥ {int(latest_data['订单平台抽成']):,}")

        st.write("**免佣卡收益**")
        pct_2 = min(1.0, latest_data['免佣卡收益'] / gmv_curr)
        st.progress(pct_2)
        st.caption(f"¥ {int(latest_data['免佣卡收益']):,}")

        st.write("**渠道活动补贴**")
        pct_3 = min(1.0, latest_data['渠道活动补贴'] / gmv_curr)
        st.progress(pct_3)
        st.caption(f"¥ {int(latest_data['渠道活动补贴']):,}")

        st.markdown(f"""
        <div style="background-color: #8B5CF6; color: white; text-align: center; padding: 10px; border-radius: 8px; margin-top: 20px;">
            🧑‍💻 {title_prefix} 司机分层
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()