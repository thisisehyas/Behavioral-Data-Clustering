from pathlib import Path

from analysis.create_summary_table import create_summary_table


def main():

    create_summary_table(
        aggregated_metrics_path=Path(
            "analysis/update_interval/aggregated/aggregated_update_interval_metrics.json"
        ),
        ari_summary_path=Path(
            "analysis/update_interval/aggregated/ari_summary_update_interval.json"
        ),
        cluster_sizes_path=Path(
            "analysis/update_interval/aggregated/aggregated_cluster_sizes.json"
        ),
        output_csv_path=Path(
            "analysis/update_interval/aggregated/update_interval_summary_table.csv"
        ),
        output_markdown_path=Path(
            "analysis/update_interval/aggregated/update_interval_summary_table.md"
        ),
        paper_output_path=Path(
            "analysis/update_interval/aggregated/paper_update_interval_summary_table.md"
        ),
        parameter_name="update_interval",
        paper_parameter_label="Update Interval",
    )


if __name__ == "__main__":
    main()