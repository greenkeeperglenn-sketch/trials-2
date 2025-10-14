"""
STRI Trial Analysis - Streamlit Web App
Single file version for quick testing
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from io import BytesIO
import base64

# Page config
st.set_page_config(
    page_title="STRI Trial Analysis",
    page_icon="🌱",
    layout="wide"
)

# Title
st.title("🌱 STRI Turf Trial Analysis Tool")
st.markdown("---")

# Initialize session state
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'statistics' not in st.session_state:
    st.session_state.statistics = None
if 'raw_data' not in st.session_state:
    st.session_state.raw_data = None
if 'treatment_map' not in st.session_state:
    st.session_state.treatment_map = {}

# Functions
def parse_trial_plan(df):
    """Extract treatment names from Trial Plan sheet"""
    treatment_map = {}
    for idx, row in df.iterrows():
        if pd.notna(row[0]) and isinstance(row[0], str) and '[' in row[0]:
            text = row[0]
            if text.startswith('[') and ']' in text:
                try:
                    num_str = text.split(']')[0].replace('[', '').strip()
                    num = int(num_str)
                    name = text.split(']')[1].strip()
                    treatment_map[num] = name
                except:
                    continue
    return treatment_map

def parse_assessment_sheet(df, sheet_name, treatment_map):
    """Parse individual assessment sheet"""
    # Find header row - look for any variation of column names
    header_row = None
    for idx in range(min(10, len(df))):  # Check first 10 rows
        row_str = ' '.join([str(val) for val in df.iloc[idx].values if pd.notna(val)])
        if 'Block' in row_str and 'Plot' in row_str and 'Treat' in row_str:
            header_row = idx
            break
    
    if header_row is None:
        st.warning(f"Could not find header row in sheet: {sheet_name}")
        return None
    
    # Get data
    headers = df.iloc[header_row].values
    data_rows = df.iloc[header_row + 1:].copy()
    data_rows.columns = headers
    
    # Find the block column (might be 'Block!', 'Block', etc.)
    block_col = None
    plot_col = None
    treat_col = None
    
    for col in data_rows.columns:
        col_str = str(col).lower()
        if 'block' in col_str:
            block_col = col
        elif 'plot' in col_str:
            plot_col = col
        elif 'treat' in col_str:
            treat_col = col
    
    if block_col is None or plot_col is None or treat_col is None:
        st.warning(f"Could not find required columns in sheet: {sheet_name}")
        return None
    
    # Filter valid rows (has block number)
    data_rows = data_rows[pd.notna(data_rows[block_col])]
    
    # Rename columns to standard names
    rename_dict = {
        block_col: 'Block',
        plot_col: 'Plot',
        treat_col: 'Treatment'
    }
    data_rows = data_rows.rename(columns=rename_dict)
    
    # Convert Treatment to int if possible
    try:
        data_rows['Treatment'] = pd.to_numeric(data_rows['Treatment'], errors='coerce')
        data_rows = data_rows[pd.notna(data_rows['Treatment'])]
        data_rows['Treatment'] = data_rows['Treatment'].astype(int)
    except:
        st.warning(f"Could not convert Treatment column in sheet: {sheet_name}")
        return None
    
    # Add treatment names
    data_rows['Treatment_Name'] = data_rows['Treatment'].map(treatment_map)
    data_rows['Assessment_Date'] = sheet_name
    
    return data_rows

def calculate_fishers_lsd(data, parameter, alpha=0.05):
    """Calculate Fisher's LSD and assign letter groups using proper algorithm"""
    try:
        treatments = sorted(data['Treatment'].unique())
        n_treatments = len(treatments)
        
        # Get group data
        groups = [data[data['Treatment'] == t][parameter].values for t in treatments]
        means = {t: data[data['Treatment'] == t][parameter].mean() for t in treatments}
        
        # Check for variation
        all_values = np.concatenate(groups)
        if len(np.unique(all_values)) == 1:
            return None, {t: 'a' for t in treatments}
        
        # Calculate MSE
        ss_within = sum([np.sum((group - means[t])**2) for t, group in zip(treatments, groups)])
        df_within = len(data) - n_treatments
        
        if df_within <= 0 or ss_within == 0:
            return None, {t: 'a' for t in treatments}
        
        mse = ss_within / df_within
        n_per_treatment = len(groups[0])
        
        if n_per_treatment == 0 or mse == 0:
            return None, {t: 'a' for t in treatments}
        
        # Calculate LSD
        t_critical = stats.t.ppf(1 - alpha/2, df_within)
        lsd = t_critical * np.sqrt(2 * mse / n_per_treatment)
        
        # Proper letter grouping algorithm
        sorted_means = sorted(means.items(), key=lambda x: x[1], reverse=True)
        letter_groups = {}
        current_letter = 0
        
        for i, (t1, mean1) in enumerate(sorted_means):
            # Find all treatments not significantly different from this one
            group_letters = set()
            
            for j, (t2, mean2) in enumerate(sorted_means):
                if abs(mean1 - mean2) <= lsd:
                    # Not significantly different
                    if t2 in letter_groups:
                        # Already has letters, add them
                        group_letters.update(letter_groups[t2])
            
            # If no existing letters, assign new one
            if not group_letters:
                group_letters.add(chr(ord('a') + current_letter))
                current_letter += 1
            
            letter_groups[t1] = ''.join(sorted(group_letters))
        
        # Clean up - ensure proper ordering
        # Highest mean should have 'a', lowest should have highest letter
        final_groups = {}
        sorted_by_mean = sorted(means.items(), key=lambda x: x[1], reverse=True)
        assigned_letters = {}
        next_letter = 0
        
        for t, mean in sorted_by_mean:
            letters = set()
            for other_t, other_mean in sorted_by_mean:
                if abs(mean - other_mean) <= lsd:
                    if other_t in assigned_letters:
                        letters.update(assigned_letters[other_t])
            
            if not letters:
                letters.add(chr(ord('a') + next_letter))
                next_letter += 1
            
            assigned_letters[t] = letters
            final_groups[t] = ''.join(sorted(letters))
        
        return lsd, final_groups
        
    except Exception as e:
        return None, {t: 'a' for t in data['Treatment'].unique()}

def analyze_data(raw_data, treatment_map, parameters, alpha=0.05):
    """Run statistical analysis"""
    all_data = pd.concat(raw_data.values(), ignore_index=True)
    assessment_dates = sorted(raw_data.keys())
    
    statistics = {}
    
    for parameter in parameters:
        if parameter not in all_data.columns:
            continue
        
        param_results = {
            'dates': assessment_dates,
            'treatments': sorted(treatment_map.keys()),
            'treatment_names': [treatment_map[t] for t in sorted(treatment_map.keys())],
            'means': {},
            'letter_groups': {},
            'statistics': {}
        }
        
        for date in assessment_dates:
            date_data = all_data[all_data['Assessment_Date'] == date].copy()
            date_data = date_data[pd.notna(date_data[parameter])]
            
            if len(date_data) == 0:
                continue
            
            # Calculate means
            means = date_data.groupby('Treatment')[parameter].mean()
            param_results['means'][date] = means.to_dict()
            
            # Run ANOVA
            treatment_groups = [
                date_data[date_data['Treatment'] == t][parameter].values 
                for t in sorted(treatment_map.keys())
                if t in date_data['Treatment'].values
            ]
            
            treatment_groups = [g for g in treatment_groups if len(g) > 0]
            
            if len(treatment_groups) > 1:
                try:
                    # Check if there's any variation in the data
                    all_values = np.concatenate(treatment_groups)
                    if len(np.unique(all_values)) == 1:
                        # All values are identical - no variation
                        param_results['statistics'][date] = {
                            'p_value': None,
                            'f_stat': None,
                            'lsd': None,
                            'significant': False
                        }
                        param_results['letter_groups'][date] = {t: 'ns' for t in means.index}
                        continue
                    
                    f_stat, p_value = stats.f_oneway(*treatment_groups)
                    
                    if p_value < alpha:
                        lsd_value, letter_groups = calculate_fishers_lsd(date_data, parameter, alpha)
                        param_results['statistics'][date] = {
                            'p_value': p_value,
                            'f_stat': f_stat,
                            'lsd': lsd_value,
                            'significant': True
                        }
                        param_results['letter_groups'][date] = letter_groups
                    else:
                        param_results['statistics'][date] = {
                            'p_value': p_value,
                            'f_stat': f_stat,
                            'lsd': None,
                            'significant': False
                        }
                        param_results['letter_groups'][date] = {t: 'ns' for t in means.index}
                        
                except Exception as e:
                    st.warning(f"Error calculating statistics for {parameter} on {date}: {str(e)}")
                    param_results['statistics'][date] = {
                        'p_value': None,
                        'f_stat': None,
                        'lsd': None,
                        'significant': False
                    }
                    param_results['letter_groups'][date] = {t: 'ns' for t in means.index}
        
        statistics[parameter] = param_results
    
    return statistics

def create_excel_output(statistics):
    """Create Excel file with statistics tables - no blank columns"""
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for parameter, stats in statistics.items():
            # Only include dates that have data
            dates_with_data = [d for d in stats['dates'] 
                             if d in stats['means'] and len(stats['means'][d]) > 0]
            
            if not dates_with_data:
                continue
            
            treatments = stats['treatment_names']
            
            # Create main table
            table_data = []
            for i, trt_num in enumerate(stats['treatments']):
                row = [treatments[i]]
                for date in dates_with_data:
                    if date in stats['means'] and trt_num in stats['means'][date]:
                        mean = stats['means'][date][trt_num]
                        
                        if (date in stats['letter_groups'] and 
                            trt_num in stats['letter_groups'][date] and
                            stats['letter_groups'][date][trt_num] != 'ns'):
                            row.append(f"{mean:.2f} {stats['letter_groups'][date][trt_num]}")
                        else:
                            row.append(f"{mean:.2f}")
                    else:
                        row.append('')
                table_data.append(row)
            
            df = pd.DataFrame(table_data, columns=['Treatment'] + dates_with_data)
            
            # Add statistical rows - matching GenStat format
            blank_row = [''] * len(df.columns)
            p_row = ['P']
            lsd_row = ['LSD']
            df_row = ['d.f.']
            cv_row = ['%c.v.']
            
            for date in dates_with_data:
                if date in stats['statistics']:
                    stat = stats['statistics'][date]
                    
                    # P-value
                    p = stat.get('p_value')
                    if p is not None:
                        if p < 0.001:
                            p_row.append('<0.001')
                        elif stat.get('significant', False):
                            p_row.append(f"{p:.3f}")
                        else:
                            p_row.append('ns')
                    else:
                        p_row.append('ns')
                    
                    # LSD
                    if stat.get('lsd'):
                        lsd_row.append(f"{stat['lsd']:.2f}")
                    else:
                        lsd_row.append('-')
                    
                    # d.f. (degrees of freedom)
                    if stat.get('df'):
                        df_row.append(str(stat['df']))
                    else:
                        df_row.append('-')
                    
                    # %c.v. (coefficient of variation)
                    if stat.get('cv'):
                        cv_row.append(f"{stat['cv']:.1f}")
                    else:
                        cv_row.append('-')
                else:
                    p_row.append('ns')
                    lsd_row.append('-')
                    df_row.append('-')
                    cv_row.append('-')
            
            stat_df = pd.DataFrame([blank_row, p_row, lsd_row, df_row, cv_row], columns=df.columns)
            final_df = pd.concat([df, stat_df], ignore_index=True)
            
            final_df.to_excel(writer, sheet_name=parameter[:31], index=False)
    
    output.seek(0)
    return output

# Sidebar
st.sidebar.header("📋 Instructions")
st.sidebar.markdown("""
1. Upload your STRI assessment Excel file
2. Click 'Analyze Data'
3. Review results
4. Download statistics tables
5. Download HTML report

**File Requirements:**
- Multiple date sheets (e.g., "03.07.25")
- Trial Plan sheet with treatments
- Standard STRI format
""")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📤 Upload Assessment File")
    uploaded_file = st.file_uploader(
        "Choose your STRI assessment Excel file",
        type=['xlsx', 'xls'],
        help="Upload the Excel file containing multiple assessment date sheets"
    )

with col2:
    st.header("⚙️ Settings")
    project_name = st.text_input("Project Name", "trial_analysis")
    alpha = st.slider("Significance Level (α)", 0.01, 0.15, 0.05, 0.01,
                     help="P-value threshold for significance. Default is 0.05 (5%). Higher values are more lenient.")

if uploaded_file is not None:
    
    with st.spinner("Loading file..."):
        try:
            # Read Excel file
            xl_file = pd.ExcelFile(uploaded_file)
            
            # Parse Trial Plan
            if 'Trial Plan' in xl_file.sheet_names:
                trial_plan = pd.read_excel(uploaded_file, sheet_name='Trial Plan', header=None)
                st.session_state.treatment_map = parse_trial_plan(trial_plan)
            else:
                st.error("No 'Trial Plan' sheet found!")
                st.stop()
            
            # Parse assessment sheets
            date_sheets = [s for s in xl_file.sheet_names 
                          if s not in ['Trial Plan', 'Sheet1']]
            
            raw_data = {}
            for sheet_name in date_sheets:
                df = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=None)
                parsed_data = parse_assessment_sheet(df, sheet_name, st.session_state.treatment_map)
                if parsed_data is not None and len(parsed_data) > 0:
                    raw_data[sheet_name] = parsed_data
            
            st.session_state.raw_data = raw_data
            
            if len(raw_data) == 0:
                st.error("No assessment data could be loaded from any sheets!")
                st.stop()
                
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            st.exception(e)
            st.stop()
    
    # Display file info
    st.success("✓ File loaded successfully!")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Treatments", len(st.session_state.treatment_map))
    with col2:
        st.metric("Assessment Dates", len(st.session_state.raw_data))
    with col3:
        total_plots = sum(len(df) for df in st.session_state.raw_data.values())
        st.metric("Total Observations", total_plots)
    with col4:
        st.metric("Replicates", 4)
    
    # Show treatments
    with st.expander("📋 View Treatments"):
        for num, name in sorted(st.session_state.treatment_map.items()):
            st.write(f"**[{num}]** {name}")
    
    # Show assessment dates
    with st.expander("📅 View Assessment Dates"):
        st.write(", ".join(sorted(st.session_state.raw_data.keys())))
    
    st.markdown("---")
    
    # Analysis button
    if st.button("📊 Analyze Data", type="primary", use_container_width=True):
        with st.spinner("Running statistical analysis..."):
            parameters = ['TQ', 'TC', '%LGC', '%DS', '%Scar', 'NDVI', 'Phyto']
            st.session_state.statistics = analyze_data(
                st.session_state.raw_data,
                st.session_state.treatment_map,
                parameters,
                alpha
            )
            st.session_state.analysis_complete = True
        
        st.success("✓ Analysis complete!")
        st.balloons()
    
    # Display results
    if st.session_state.analysis_complete and st.session_state.statistics:
        st.markdown("---")
        st.header("📊 Results Summary")
        
        # Summary metrics
        sig_count = 0
        for param, stats in st.session_state.statistics.items():
            for date, stat_info in stats['statistics'].items():
                if stat_info.get('significant', False):
                    sig_count += 1
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Parameters Analyzed", len(st.session_state.statistics))
        with col2:
            st.metric("Significant Results", sig_count)
        
        # Parameter results
        for param, stats in st.session_state.statistics.items():
            with st.expander(f"📈 {param} - Results"):
                
                # Find significant dates
                sig_dates = [date for date, stat_info in stats['statistics'].items() 
                           if stat_info.get('significant', False)]
                
                if sig_dates:
                    st.success(f"✓ Significant treatment effects detected on {len(sig_dates)} dates")
                    
                    # Show results for first significant date
                    date = sig_dates[0]
                    st.subheader(f"Results for {date}")
                    
                    # Create results table
                    results_data = []
                    for i, trt_num in enumerate(stats['treatments']):
                        if date in stats['means'] and trt_num in stats['means'][date]:
                            mean = stats['means'][date][trt_num]
                            letter = stats['letter_groups'][date].get(trt_num, '')
                            results_data.append({
                                'Treatment': stats['treatment_names'][i],
                                'Mean': f"{mean:.2f}",
                                'Group': letter
                            })
                    
                    results_df = pd.DataFrame(results_data)
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Show statistics
                    st.write(f"**P-value:** {stats['statistics'][date]['p_value']:.4f}")
                    st.write(f"**LSD:** {stats['statistics'][date]['lsd']:.4f}")
                else:
                    st.info("No significant treatment effects detected")
        
        st.markdown("---")
        
        # Download section
        st.header("📥 Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Excel download
            excel_file = create_excel_output(st.session_state.statistics)
            st.download_button(
                label="📊 Download Excel Statistics Tables",
                data=excel_file,
                file_name=f"{project_name}_statistics.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
        with col2:
            # HTML report placeholder
            st.info("HTML report generation available in full version")

else:
    st.info("👆 Upload your STRI assessment file to begin")
    
    # Example section
    with st.expander("📖 Example File Format"):
        st.markdown("""
        Your Excel file should contain:
        
        **Assessment Date Sheets** (e.g., "03.07.25"):
        - Row 1-6: Metadata (Trial Code, Date, Area, etc.)
        - Row 7: Headers (Block! | Plot! | Treat1! | TQ | TC | %LGC | %DS | %Scar | NDVI | Phyto)
        - Row 8+: Data (36 plots)
        
        **Trial Plan Sheet**:
        - Treatment mapping:
          ```
          [1] Untreated
          [2] Ryder
          [3] GM Liquid Effect Fe
          ...
          ```
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 12px;'>
    STRI Trial Analysis Tool | Built with Streamlit | Statistical analysis using Python scipy
</div>
""", unsafe_allow_html=True)
