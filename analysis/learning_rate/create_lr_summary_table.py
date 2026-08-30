from pathlib import Path

from analysis.create_summary_table import create_summary_table


def main():

    create_summary_table(
        aggregated_metrics_path=Path(
            "analysis/learning_rate/aggregated/aggregated_lr_metrics.json"
        ),
        ari_summary_path=Path(
            "analysis/learning_rate/aggregated/ari_summary.json"
        ),
        cluster_sizes_path=Path(
            "analysis/learning_rate/aggregated/aggregated_cluster_sizes.json"
        ),
        output_csv_path=Path(
            "analysis/learning_rate/aggregated/learning_rate_summary_table.csv"
        ),
        output_markdown_path=Path(
            "analysis/learning_rate/aggregated/learning_rate_summary_table.md"
        ),
        paper_output_path=Path(
            "analysis/learning_rate/aggregated/paper_learning_rate_summary_table.md"
        ),
        parameter_name="learning_rate",
        paper_parameter_label="Learning Rate",
    )


if __name__ == "__main__":
    main()