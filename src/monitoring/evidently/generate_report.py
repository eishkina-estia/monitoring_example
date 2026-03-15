import os

from src.monitoring.evidently.evidently_utils import (
    build_reference_dataframe,
    load_current_dataframe,
    generate_drift_report
)

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")

from src import common as common
SCENARIOS_DIR = common.CONFIG["paths"]["monitoring"]["drift_scenarios_dir"]
SCENARIO_FILES = common.CONFIG["monitoring"]["drift_scenarios_files"]

def generate_report_for_file(csv_path: str, reference_df, feature_columns):

    scenario_name = os.path.splitext(os.path.basename(csv_path))[0]
    output_path = os.path.join(REPORTS_DIR, f"{scenario_name}_report.html")

    current_df = load_current_dataframe(csv_path, feature_columns)
    generate_drift_report(reference_df, current_df, output_path)


if __name__ == "__main__":

    print("Generate evidently reports...")

    reference_df, feature_columns = build_reference_dataframe()

    for filename in SCENARIO_FILES:
        csv_path = os.path.join(SCENARIOS_DIR, filename)

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"File not found: {csv_path}")

        print(f"Processing: {filename}")
        generate_report_for_file(csv_path, reference_df, feature_columns)

    print("All reports generated successfully.")