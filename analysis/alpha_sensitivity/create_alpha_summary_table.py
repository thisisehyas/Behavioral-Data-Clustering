from pathlib import Path

from analysis.create_summary_table import create_summary_table


def main():

    create_summary_table(
        aggregated_metrics_path=Path(
            "analysis/alpha_sensitivity/aggregated/aggregated_alpha_metrics.json"
        ),
        ari_summary_path=Path(
            "analysis/alpha_sensitivity/aggregated/ari_summary_alpha.json"
        ),
        cluster_sizes_path=Path(
            "analysis/alpha_sensitivity/aggregated/aggregated_cluster_sizes.json"
        ),
        output_csv_path=Path(
            "analysis/alpha_sensitivity/aggregated/alpha_summary_table.csv"
        ),
        output_markdown_path=Path(
            "analysis/alpha_sensitivity/aggregated/alpha_summary_table.md"
        ),
        paper_output_path=Path(
            "analysis/alpha_sensitivity/aggregated/paper_alpha_summary_table.md"
        ),
        parameter_name="alpha",
        paper_parameter_label="α",
    )


if __name__ == "__main__":
    main()