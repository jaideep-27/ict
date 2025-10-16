import streamlit as st
import pandas as pd
import plotly.express as px

# -----------------------------------------
# Page Config
# -----------------------------------------
st.set_page_config(
    page_title="DataSense — CSV Explorer",
    page_icon="📊",
    layout="wide"
)

# -----------------------------------------
# Custom Clean Dark Theme CSS
# -----------------------------------------
st.markdown("""
<style>
/* General layout */
.main {
    background-color: #0e1117;
    color: #e6edf3;
    font-family: "Inter", sans-serif;
    padding: 1.5rem 2rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(18, 21, 27, 0.95);
    backdrop-filter: blur(10px);
    border-right: 1px solid #30363d;
}

/* Headings */
h1, h2, h3 {
    color: #f0f6fc;
    font-weight: 600;
}
h1 {
    margin-bottom: 1rem;
}
h2, h3 {
    color: #c9d1d9;
}

/* Metrics */
div[data-testid="stMetricValue"] {
    color: #58a6ff;
    font-size: 1.7rem;
    font-weight: 700;
}
div[data-testid="stMetricLabel"] {
    color: #9da7b2;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background-color: rgba(22, 27, 34, 0.8);
    border-radius: 8px;
    padding: 0.4rem;
}
.stTabs [data-baseweb="tab"] {
    color: #8b949e;
    border-radius: 6px;
    transition: all 0.2s ease;
}
.stTabs [data-baseweb="tab"]:hover {
    background-color: rgba(88, 166, 255, 0.15);
    color: #58a6ff;
}
.stTabs [aria-selected="true"] {
    background-color: #161b22;
    color: #ffffff;
    font-weight: 600;
    border-bottom: none;
}

/* Buttons */
.stDownloadButton button {
    background: #1f6feb;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 0.6rem 1.2rem;
    font-weight: 600;
}
.stDownloadButton button:hover {
    background: #388bfd;
}

/* Dataframe styling */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    background-color: rgba(22,27,34,0.6);
}

/* Info & captions */
p, span, div, li {
    color: #e6edf3;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------
# Header
# -----------------------------------------
st.title("📊 DataSense — CSV Explorer")
st.caption("Explore, visualize, and download your CSV data in a clean, modern dashboard.")

# -----------------------------------------
# Sidebar Upload
# -----------------------------------------
with st.sidebar:
    st.header("📁 Upload CSV")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    st.markdown("---")

    if uploaded_file:
        num_rows = st.slider("Rows to display", 5, 100, 10)
        show_stats = st.checkbox("Show summary statistics", True)
        show_viz = st.checkbox("Enable visualizations", True)
        st.caption("Supports CSV up to 200MB.")

# -----------------------------------------
# Main App
# -----------------------------------------
if uploaded_file:
    df = pd.read_csv(uploaded_file, low_memory=False)

    # Data Preview
    st.subheader("🧩 Data Preview")
    st.caption(f"Showing first {num_rows} of {len(df)} rows")
    st.dataframe(df.head(num_rows), use_container_width=True, height=400)

    # Overview Metrics
    st.markdown("---")
    st.subheader("📈 Dataset Overview")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Rows", f"{df.shape[0]:,}")
    col2.metric("Columns", df.shape[1])
    col3.metric("Memory (KB)", f"{df.memory_usage(deep=True).sum() / 1024:.1f}")
    col4.metric("Numeric Cols", len(df.select_dtypes('number').columns))
    col5.metric("Text Cols", len(df.select_dtypes('object').columns))

    # Column Details
    st.markdown("---")
    st.subheader("📋 Column Details")
    col_info = pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes.astype(str),
        'Non-Null': df.count(),
        'Null': df.isnull().sum(),
        'Unique': df.nunique()
    })
    st.dataframe(col_info, use_container_width=True, height=300)

    # Stats
    if show_stats:
        st.markdown("---")
        st.subheader("📊 Statistical Summary")
        
        tab_num, tab_cat = st.tabs(["Numeric Columns", "Categorical Columns"])
        
        with tab_num:
            numeric_cols = df.select_dtypes('number').columns.tolist()
            if numeric_cols:
                st.dataframe(df[numeric_cols].describe().T, use_container_width=True, height=400)
            else:
                st.info("No numeric columns in dataset")
        
        with tab_cat:
            categorical_cols = df.select_dtypes('object').columns.tolist()
            if categorical_cols:
                cat_summary = []
                for col in categorical_cols:
                    cat_summary.append({
                        'Column': col,
                        'Unique Values': df[col].nunique(),
                        'Most Common': df[col].mode()[0] if len(df[col].mode()) > 0 else 'N/A',
                        'Most Common Count': df[col].value_counts().iloc[0] if len(df[col]) > 0 else 0,
                        'Missing': df[col].isnull().sum()
                    })
                st.dataframe(pd.DataFrame(cat_summary), use_container_width=True, height=400)
            else:
                st.info("No categorical columns in dataset")

    # Visualizations
    if show_viz:
        st.markdown("---")
        st.subheader("🎨 Visual Insights")

        numeric_cols = df.select_dtypes('number').columns.tolist()
        categorical_cols = df.select_dtypes('object').columns.tolist()

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Distribution", "Branch Placements", "Sunburst & Treemap", "Scatter", "Correlation", "Category Analysis"])

        # Distribution - Histogram for numeric columns
        with tab1:
            if numeric_cols:
                col = st.selectbox("Choose numeric column", numeric_cols, key="hist_col")
                fig = px.histogram(df, x=col, nbins=30, color_discrete_sequence=['#58a6ff'],
                                 title=f"Distribution of {col}")
                fig.update_layout(template="plotly_dark", height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Show statistics for selected column
                col_stats = df[col].describe()
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Mean", f"{col_stats['mean']:.2f}")
                c2.metric("Median", f"{col_stats['50%']:.2f}")
                c3.metric("Std Dev", f"{col_stats['std']:.2f}")
                c4.metric("Range", f"{col_stats['max'] - col_stats['min']:.2f}")
            else:
                st.info("No numeric columns found for histogram.")

        # Branch-wise Placements Histogram
        with tab2:
            # Try to detect branch/department column
            branch_col = None
            placement_col = None
            
            # Look for common column names
            for col in categorical_cols:
                if any(keyword in col.lower() for keyword in ['branch', 'department', 'dept', 'stream', 'course']):
                    branch_col = col
                    break
            
            if branch_col:
                st.markdown(f"#### Branch-wise Analysis (using '{branch_col}' column)")
                
                # Count placements by branch
                branch_counts = df[branch_col].value_counts().reset_index()
                branch_counts.columns = [branch_col, 'Count']
                
                # Create histogram/bar chart
                fig = px.bar(branch_counts, x=branch_col, y='Count', 
                           title=f"Distribution by {branch_col}",
                           color='Count',
                           color_continuous_scale='Blues')
                fig.update_layout(template="plotly_dark", height=450, showlegend=False)
                fig.update_xaxes(tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
                
                # Show top branches
                st.markdown("##### Top Categories")
                st.dataframe(branch_counts.head(10), use_container_width=True, hide_index=True)
                
            else:
                # Manual selection
                st.markdown("#### Custom Branch/Category Analysis")
                if categorical_cols:
                    selected_col = st.selectbox("Select column for branch/category analysis", categorical_cols, key="branch_select")
                    
                    value_counts = df[selected_col].value_counts().reset_index()
                    value_counts.columns = [selected_col, 'Count']
                    
                    fig = px.bar(value_counts.head(20), x=selected_col, y='Count',
                               title=f"Top 20 {selected_col} Distribution",
                               color='Count',
                               color_continuous_scale='Blues')
                    fig.update_layout(template="plotly_dark", height=450, showlegend=False)
                    fig.update_xaxes(tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.dataframe(value_counts.head(15), use_container_width=True, hide_index=True)
                else:
                    st.info("No categorical columns found. Upload a CSV with branch/category data.")

        # Sunburst & Treemap
        with tab3:
            st.markdown("#### Hierarchical Visualizations")
            
            if categorical_cols:
                # Let user select columns for hierarchy
                st.markdown("##### Configure Hierarchy")
                
                # Try to auto-detect useful columns
                primary_col = None
                secondary_col = None
                
                # Look for branch/department
                for col in categorical_cols:
                    if any(keyword in col.lower() for keyword in ['branch', 'department', 'dept', 'stream', 'course']):
                        primary_col = col
                        break
                
                # User selection
                col1, col2 = st.columns(2)
                with col1:
                    level1 = st.selectbox("Primary Category", categorical_cols, 
                                         index=categorical_cols.index(primary_col) if primary_col else 0,
                                         key="sunburst_level1")
                with col2:
                    remaining_cols = [c for c in categorical_cols if c != level1]
                    if remaining_cols:
                        level2 = st.selectbox("Secondary Category (optional)", ["None"] + remaining_cols, key="sunburst_level2")
                    else:
                        level2 = "None"
                
                # Prepare data
                if level2 != "None":
                    # Two-level hierarchy
                    hierarchy_data = df.groupby([level1, level2]).size().reset_index(name='Count')
                    
                    # Sunburst Chart
                    st.markdown("##### 🌅 Sunburst Chart")
                    fig_sun = px.sunburst(hierarchy_data, path=[level1, level2], values='Count',
                                         color='Count', color_continuous_scale='Blues',
                                         title=f"Hierarchical Distribution: {level1} → {level2}")
                    fig_sun.update_layout(template="plotly_dark", height=500)
                    st.plotly_chart(fig_sun, use_container_width=True)
                    
                    # Treemap
                    st.markdown("##### 🗺️ Treemap")
                    fig_tree = px.treemap(hierarchy_data, path=[level1, level2], values='Count',
                                         color='Count', color_continuous_scale='Teal',
                                         title=f"Treemap: {level1} → {level2}")
                    fig_tree.update_layout(template="plotly_dark", height=500)
                    st.plotly_chart(fig_tree, use_container_width=True)
                    
                else:
                    # Single-level hierarchy
                    hierarchy_data = df[level1].value_counts().reset_index()
                    hierarchy_data.columns = [level1, 'Count']
                    
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.markdown("##### 🌅 Sunburst Chart")
                        fig_sun = px.sunburst(hierarchy_data, path=[level1], values='Count',
                                             color='Count', color_continuous_scale='Blues',
                                             title=f"Distribution by {level1}")
                        fig_sun.update_layout(template="plotly_dark", height=450)
                        st.plotly_chart(fig_sun, use_container_width=True)
                    
                    with col_b:
                        st.markdown("##### 🗺️ Treemap")
                        fig_tree = px.treemap(hierarchy_data, path=[level1], values='Count',
                                             color='Count', color_continuous_scale='Teal',
                                             title=f"Treemap: {level1}")
                        fig_tree.update_layout(template="plotly_dark", height=450)
                        st.plotly_chart(fig_tree, use_container_width=True)
                
                # Show summary table
                st.markdown("##### 📊 Summary Table")
                if level2 != "None":
                    summary = df.groupby([level1, level2]).size().reset_index(name='Count')
                    summary = summary.sort_values('Count', ascending=False)
                    st.dataframe(summary.head(20), use_container_width=True, hide_index=True)
                else:
                    summary = df[level1].value_counts().reset_index()
                    summary.columns = [level1, 'Count']
                    summary['Percentage'] = (summary['Count'] / summary['Count'].sum() * 100).round(2).astype(str) + '%'
                    st.dataframe(summary.head(20), use_container_width=True, hide_index=True)
                    
            else:
                st.info("No categorical columns available for hierarchical visualization.")

        # Scatter Plot
        with tab4:
            if len(numeric_cols) >= 2:
                c1, c2 = st.columns(2)
                with c1:
                    x = st.selectbox("X-axis", numeric_cols, key="x_axis")
                with c2:
                    y = st.selectbox("Y-axis", numeric_cols, index=min(1, len(numeric_cols)-1), key="y_axis")
                
                # Optional: color by categorical column
                color_col = None
                if categorical_cols:
                    use_color = st.checkbox("Color by category", key="use_color")
                    if use_color:
                        color_col = st.selectbox("Select category column", categorical_cols, key="color_col")
                
                if color_col:
                    fig = px.scatter(df, x=x, y=y, color=color_col, 
                                   title=f"{y} vs {x} (colored by {color_col})")
                else:
                    fig = px.scatter(df, x=x, y=y, color_discrete_sequence=['#58a6ff'],
                                   title=f"{y} vs {x}")
                
                fig.update_layout(template="plotly_dark", height=450)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Need at least two numeric columns for scatter plot.")

        # Correlation Matrix
        with tab5:
            if len(numeric_cols) >= 2:
                corr = df[numeric_cols].corr()
                fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                              title="Correlation Heatmap", aspect="auto")
                fig.update_layout(template="plotly_dark", height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Show strongest correlations
                st.markdown("##### Strongest Correlations")
                corr_pairs = corr.unstack()
                corr_pairs = corr_pairs[corr_pairs != 1.0]
                top_corr = corr_pairs.abs().sort_values(ascending=False).head(10)
                
                corr_df = pd.DataFrame({
                    'Feature Pair': [f"{pair[0]} ↔ {pair[1]}" for pair in top_corr.index],
                    'Correlation': [f"{corr_pairs[pair]:.3f}" for pair in top_corr.index]
                })
                st.dataframe(corr_df, use_container_width=True, hide_index=True)
            else:
                st.info("Need at least two numeric columns for correlation analysis.")

        # Category Analysis - Pie and Bar charts for categorical data
        with tab6:
            if categorical_cols:
                st.markdown("#### Categorical Data Analysis")
                
                selected_cat = st.selectbox("Select categorical column", categorical_cols, key="cat_analysis")
                
                # Get value counts
                value_counts = df[selected_cat].value_counts().head(15)
                
                # Create two columns for pie and bar
                c1, c2 = st.columns(2)
                
                with c1:
                    st.markdown("##### Pie Chart")
                    fig_pie = px.pie(values=value_counts.values, names=value_counts.index,
                                   title=f"Distribution of {selected_cat}",
                                   color_discrete_sequence=px.colors.sequential.Blues_r)
                    fig_pie.update_layout(template="plotly_dark", height=400)
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with c2:
                    st.markdown("##### Bar Chart")
                    fig_bar = px.bar(x=value_counts.index, y=value_counts.values,
                                   labels={'x': selected_cat, 'y': 'Count'},
                                   color=value_counts.values,
                                   color_continuous_scale='Blues')
                    fig_bar.update_layout(template="plotly_dark", height=400, showlegend=False)
                    fig_bar.update_xaxes(tickangle=-45)
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                # Show value counts table
                st.markdown("##### Value Counts")
                counts_df = pd.DataFrame({
                    selected_cat: value_counts.index,
                    'Count': value_counts.values,
                    'Percentage': [f"{(v/len(df)*100):.2f}%" for v in value_counts.values]
                })
                st.dataframe(counts_df, use_container_width=True, hide_index=True)
                
            else:
                st.info("No categorical columns found for analysis.")

    # Download CSV
    st.markdown("---")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download CSV", csv, "processed_data.csv", "text/csv")

else:
    st.info("👆 Upload a CSV file to begin exploration.")
    st.markdown("---")
    st.write("""
    **Features**
    - Clean matte dark interface  
    - Instant data preview  
    - Smart numeric & categorical analysis  
    - Pie, scatter, histogram & correlation charts  
    - One-click CSV export  
    """)
