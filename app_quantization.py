import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="FICO Score Strategic Quantization Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎯 FICO Score Strategic Quantization & Rating Engine")
st.markdown("""
This utility implements high-performance **Dynamic Programming** to find the mathematically optimal bucket boundaries 
for FICO scores. It transforms continuous credit profiles into optimal categories required by discrete risk models.
""")

# ==========================================
# 2. CACHED DATA LOADING & AGGREGATION
# ==========================================
@st.cache_data
def load_and_aggregate_data():
    # Load dataset
    df = pd.read_csv('Task 3 and 4_Loan_Data.csv')
    
    # Sort and aggregate by unique FICO scores to minimize DP complexity
    fico_raw = df['fico_score'].values
    default_raw = df['default'].values
    
    unique_ficos, counts = np.unique(fico_raw, return_counts=True)
    defaults_per_fico = np.array([np.sum(default_raw[fico_raw == f]) for f in unique_ficos])
    
    return df, unique_ficos, counts, defaults_per_fico

try:
    df, unique_ficos, counts, defaults_per_fico = load_and_aggregate_data()
except FileNotFoundError:
    st.error("❌ 'Task 3 and 4_Loan_Data.csv' not found. Please ensure it is in the active directory.")
    st.stop()

# ==========================================
# 3. DYNAMIC PROGRAMMING OPTIMIZATION CORES
# ==========================================
def optimize_log_likelihood(unique_ficos, counts, defaults, num_buckets):
    N = len(unique_ficos)
    pref_n = np.zeros(N + 1, dtype=int)
    pref_k = np.zeros(N + 1, dtype=int)
    for i in range(N):
        pref_n[i+1] = pref_n[i] + counts[i]
        pref_k[i+1] = pref_k[i] + defaults[i]
        
    ll_matrix = np.full((N, N), -np.inf)
    for i in range(N):
        for j in range(i, N):
            n = pref_n[j+1] - pref_n[i]
            k = pref_k[j+1] - pref_k[i]
            if n > 0:
                p = k / n
                term1 = k * np.log(p) if k > 0 else 0
                term2 = (n - k) * np.log(1 - p) if (n - k) > 0 else 0
                ll_matrix[i, j] = term1 + term2
                
    dp = np.full((num_buckets + 1, N), -np.inf)
    parent = np.full((num_buckets + 1, N), -1, dtype=int)
    
    for i in range(N):
        dp[1, i] = ll_matrix[0, i]
        
    for b in range(2, num_buckets + 1):
        for i in range(N):
            for j in range(b - 2, i):
                val = dp[b-1, j] + ll_matrix[j+1, i]
                if val > dp[b, i]:
                    dp[b, i] = val
                    parent[b, i] = j
                    
    boundaries = []
    curr = N - 1
    for b in range(num_buckets, 1, -1):
        idx = parent[b, curr]
        boundaries.append(int(unique_ficos[idx+1]))
        curr = idx
    boundaries.reverse()
    return boundaries

def optimize_mse(unique_ficos, counts, num_buckets):
    N = len(unique_ficos)
    mse_matrix = np.full((N, N), np.inf)
    for i in range(N):
        f_sub, c_sub = [], []
        for k in range(i, N):
            f_sub.append(unique_ficos[k])
            c_sub.append(counts[k])
            total_count = sum(c_sub)
            total_sum = sum(f * c for f, c in zip(f_sub, c_sub))
            mean = total_sum / total_count
            sq_err = sum(c * (f - mean)**2 for f, c in zip(f_sub, c_sub))
            mse_matrix[i, k] = sq_err
            
    dp = np.full((num_buckets + 1, N), np.inf)
    parent = np.full((num_buckets + 1, N), -1, dtype=int)
    
    for i in range(N):
        dp[1, i] = mse_matrix[0, i]
        
    for b in range(2, num_buckets + 1):
        for i in range(N):
            for j in range(b - 2, i):
                val = dp[b-1, j] + mse_matrix[j+1, i]
                if val < dp[b, i]:
                    dp[b, i] = val
                    parent[b, i] = j
                    
    boundaries = []
    curr = N - 1
    for b in range(num_buckets, 1, -1):
        idx = parent[b, curr]
        boundaries.append(int(unique_ficos[idx+1]))
        curr = idx
    boundaries.reverse()
    return boundaries

# ==========================================
# 4. SIDEBAR CONTROLS
# ==========================================
st.sidebar.header("⚙️ Optimization Parameters")
num_buckets = st.sidebar.slider("Target Number of Rating Buckets", min_value=2, max_value=10, value=5)
objective_type = st.sidebar.radio(
    "Optimization Objective Strategy",
    options=["Log-Likelihood Maximization", "Mean Squared Error (MSE) Minimization"],
    help="Log-Likelihood maximizes risk segmentation based on default histories; MSE optimizes based on the density distribution of FICO scores."
)

# Run Selected DP Optimization
if objective_type == "Log-Likelihood Maximization":
    optimal_boundaries = optimize_log_likelihood(unique_ficos, counts, defaults_per_fico, num_buckets)
else:
    optimal_boundaries = optimize_mse(unique_ficos, counts, num_buckets)

# Build Complete Chronological Rating Mapping Array
# Note: Task states lower rating maps to a better credit score.
# Therefore: Highest FICO score bucket -> Rating 1, Lowest FICO bucket -> Max Rating.
all_bounds = [int(unique_ficos[0])] + optimal_boundaries + [int(unique_ficos[-1] + 1)]
map_records = []

for i in range(len(all_bounds) - 1):
    low = all_bounds[i]
    high = all_bounds[i+1] - 1
    mask = (unique_ficos >= low) & (unique_ficos <= high)
    
    n_rec = int(np.sum(counts[mask]))
    k_def = int(np.sum(defaults_per_fico[mask]))
    pd_val = k_def / n_rec if n_rec > 0 else 0.0
    
    map_records.append({
        "Lower Bound": low,
        "Upper Bound": high,
        "Borrower Count": n_rec,
        "Defaults Count": k_def,
        "Probability of Default (PD)": pd_val
    })

# Sort so that highest FICO ranges receive the lowest (best) credit rating ranking
map_df = pd.DataFrame(map_records)
map_df = map_df.sort_values(by="Lower Bound", ascending=False).reset_index(drop=True)
map_df.insert(0, "Assigned Credit Rating", range(1, num_buckets + 1))

# ==========================================
# 5. MAIN CONTENT TAB ARCHITECTURE
# ==========================================
tab1, tab2 = st.tabs(["📋 Optimal Rating Map", "📊 Portfolio Statistical Analytics"])

with tab1:
    st.subheader("🏁 Evaluated Optimal Scorecard Matrix")
    st.caption("As instructed, a lower assigned numerical rating implies a stronger credit file (lower Probability of Default).")
    
    # Beautifully display the map dataframe
    st.dataframe(
        map_df.style.format({
            "Probability of Default (PD)": "{:.2%}",
            "Borrower Count": "{:,}",
            "Defaults Count": "{:,}"
        }),
        use_container_width=True
    )
    
    st.markdown("---")
    st.subheader("🔍 Production Live Map Query Tool")
    c1, c2 = st.columns([1, 2])
    with c1:
        query_score = st.number_input("Enter Borrower FICO Score:", min_value=300, max_value=850, value=620)
    with c2:
        # Resolve rating map entry location
        matched_row = map_df[(query_score >= map_df["Lower Bound"]) & (query_score <= map_df["Upper Bound"])]
        if not matched_row.empty:
            assigned_r = matched_row["Assigned Credit Rating"].values[0]
            assigned_pd = matched_row["Probability of Default (PD)"].values[0]
            st.metric(
                label=f"Assigned Categorical Feature Value (Rating Group)",
                value=f"Rating Grade: {assigned_r}",
                delta=f"Expected Bucket PD: {assigned_pd:.2%}",
                delta_color="inverse"
            )
        else:
            st.warning("⚠️ Entered FICO score falls outside the observed data parameters.")

with tab2:
    st.subheader("📊 Quantization Splitting Visualizations")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # Left Plot: Population Histogram with Boundaries
    ax1.hist(df['fico_score'], bins=40, color='darkblue', alpha=0.7, edgecolor='k', label='Borrower Count')
    for b_val in optimal_boundaries:
        ax1.axvline(x=b_val, color='red', linestyle='--', linewidth=1.5, zorder=5)
    ax1.set_title("FICO Score Population Distribution & Boundaries")
    ax1.set_xlabel("FICO Score")
    ax1.set_ylabel("Borrower Records")
    ax1.grid(True, alpha=0.15)
    
    # Right Plot: Monotonic Credit Rating Scale Check
    sorted_by_rating = map_df.sort_values("Assigned Credit Rating")
    ax2.bar(
        sorted_by_rating["Assigned Credit Rating"].astype(str),
        sorted_by_rating["Probability of Default (PD)"] * 100,
        color='crimson', alpha=0.8, edgecolor='k'
    )
    ax2.set_title("Probability of Default (PD) per Assigned Rating")
    ax2.set_xlabel("Assigned Credit Rating (Lower = Better Credit)")
    ax2.set_ylabel("Empirical Default Rate (%)")
    ax2.grid(True, alpha=0.15)
    
    plt.tight_layout()
    st.pyplot(fig)