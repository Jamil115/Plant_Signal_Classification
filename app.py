import streamlit as st
import pandas as pd
from processing import run_experiment  # Make sure this file exists

# Set up page config
st.set_page_config(page_title="Data Experiment App", layout="wide")

# Initialize session state for navigation and experiment state
if 'section' not in st.session_state:
    st.session_state.section = "Run Experiment"
if "processing" not in st.session_state:
    st.session_state.processing = False
if "completed" not in st.session_state:
    st.session_state.completed = False
if "result" not in st.session_state:
    st.session_state.result = None

# Navigation buttons using columns
spacer1, col1, spacer2, col2, spacer3, col3, spacer4 = st.columns([2, 2, 0.3, 2, 0.3, 2, 2])

with col1:
    if st.button("Run Experiment", use_container_width=True):
        st.session_state.section = "Run Experiment"
with col2:
    if st.button("Experiment Result", use_container_width=True):
        st.session_state.section = "Experiment Result"
with col3:
    if st.button("Theoretical Analytics", use_container_width=True):
        st.session_state.section = "Theoretical Analytics"


# Section: Run Experiment
if st.session_state.section == "Run Experiment":
    st.title("Run Experiment")

    st.write("### Upload a dataset")
    uploaded_file = st.file_uploader("", type=["csv"], key="file_uploader")
    st.write("")  # spacer

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.raw_data = df.copy()
        st.success(f"File `{uploaded_file.name}` successfully uploaded!")
        st.write(f"### Preview of `{uploaded_file.name}`:")
        st.dataframe(df.head())

        # Show Run button
        if not st.session_state.processing and not st.session_state.completed:
            if st.button("Run Experiment"):
                st.session_state.processing = True
                st.rerun()

        # Show spinner while processing with live status update
        elif st.session_state.processing:

            if st.button("Reset"):
                st.session_state.processing = False
                st.session_state.completed = False
                st.session_state.result = None
                st.rerun()

            st.info("Processing dataset... Please wait.")

            status_placeholder = st.empty()  # placeholder for live status text

            def update_status(message):
                status_placeholder.markdown(f"**{message}**")

            with st.spinner("Running experiment..."):
                # Pass the update_status callback into your run_experiment function
                result = run_experiment(df, update_status=update_status)

            # Save result and update state
            st.session_state.result = result
            st.session_state.processing = False
            st.session_state.completed = True
            st.rerun()

        # After completion
        elif st.session_state.completed:

            if st.button("Reset"):
                st.session_state.processing = False
                st.session_state.completed = False
                st.session_state.result = None
                st.rerun()

            st.success("Dataset processing completed successfully!")
            st.info("✔ Data Cleaning Completed")
            st.info("✔ Features Extracted")
            st.info("✔ Ready for Analysis")
            st.success("You can now view the result in the 'Experiment Result' section.")




# Section: Experiment Result
elif st.session_state.section == "Experiment Result":
    st.title("Experiment Result")

    if st.session_state.result is None:
        st.warning("No experiment result found. Please run an experiment first.")
    else:
        result = st.session_state.result

        # Tabs: Model → Visualization → Raw Data
        tab1, tab2, tab3 = st.tabs(["Model Performance", "Visualizations", "Raw Dataset"])


        # Tab 1: Model Performance
        with tab1:
            st.subheader("Model Performance Summary")
            for model_name, res in result["models"].items():
                st.markdown(f"### {model_name}")
                st.write(f"**Best Accuracy (CV):** `{res['best_score']:.3f}`")
                st.write(f"**Test Accuracy:** `{res['test_accuracy']:.3f}`")
                with st.expander("Best Parameters"):
                    st.json(res["best_params"])
                with st.expander("Classification Report"):
                    st.json(res["report"])


        # Tab 2: Visualizations
        with tab2:
            st.subheader("Visualizations")

            if "class_dist_before" in result["plots"]:
                st.markdown("#### Class Distribution Before SMOTE")
                st.pyplot(result["plots"]["class_dist_before"])

            if "class_dist_after" in result["plots"]:
                st.markdown("#### Class Distribution After SMOTE")
                st.pyplot(result["plots"]["class_dist_after"])

            if "boxplot_before" in result["plots"]:
                st.markdown("#### Boxplot Before Outlier Removal")
                st.pyplot(result["plots"]["boxplot_before"])

            if "boxplot_after" in result["plots"]:
                st.markdown("#### Boxplot After Outlier Removal")
                st.pyplot(result["plots"]["boxplot_after"])

            if "correlation_heatmap" in result["plots"]:
                st.markdown("#### Correlation Heatmap")
                st.pyplot(result["plots"]["correlation_heatmap"])

            if "conf_matrix_all" in result["plots"]:
                st.markdown("#### Confusion Matrices (All Models)")
                st.pyplot(result["plots"]["conf_matrix_all"])


        # Tab 3: Raw Dataset
        with tab3:
            st.subheader("Uploaded Raw Dataset")
            if "raw_data" in st.session_state and st.session_state.raw_data is not None:
                st.dataframe(st.session_state.raw_data)
            else:
                st.info("ℹDataset not found. Please upload it from the 'Run Experiment' section.")



# Section: Theoretical Analytics
elif st.session_state.section == "Theoretical Analytics":
    st.title("Theoretical Analytics")
    st.write("This section will contain theoretical or analytical summaries.")
