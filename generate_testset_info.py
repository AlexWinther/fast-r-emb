#!/usr/bin/env python3
"""
Script to generate a table with useful information about the testsets used in this project.
The table includes:
- Number of tests in each test suite
- Number of faults
- Number of tests revealing those faults
- Average tests per fault
- Percentage of tests revealing faults
"""

import sys
from pathlib import Path

# Add src to path to import local modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils import load_test_suite, SIR, D4J
import pandas as pd
from rich.console import Console
from rich.table import Table
from transformers import AutoTokenizer

# Initialize CodeT5+ tokenizer for token counting
tokenizer = AutoTokenizer.from_pretrained(
    "Salesforce/codet5p-220m", trust_remote_code=True
)


def get_testset_info():
    """Load all test suites and compute metrics."""
    data = []

    # Combine SIR and D4J suites
    all_suites = SIR + D4J

    for program, version in all_suites:
        try:
            # Load the test suite
            test_suite = load_test_suite(program, version)

            # Calculate metrics
            num_tests = len(test_suite.test_cases)

            # Compute lengths
            char_lengths = [len(tc) for tc in test_suite.test_cases]
            word_lengths = [len(tc.split()) for tc in test_suite.test_cases]
            token_lengths = [
                len(tokenizer.encode(tc, add_special_tokens=True))
                for tc in test_suite.test_cases
            ]

            avg_char_len = sum(char_lengths) / num_tests if num_tests > 0 else 0
            avg_word_count = sum(word_lengths) / num_tests if num_tests > 0 else 0
            avg_token_count = sum(token_lengths) / num_tests if num_tests > 0 else 0
            min_token_count = min(token_lengths) if token_lengths else 0
            max_token_count = max(token_lengths) if token_lengths else 0

            # Get all unique faults
            all_faults = set()
            for faults in test_suite.fault_matrix.values():
                all_faults.update(faults)
            num_faults = len(all_faults)

            # Count tests that reveal at least one fault
            num_fault_revealing_tests = len(
                [
                    tc_id
                    for tc_id in test_suite.fault_matrix
                    if test_suite.fault_matrix[tc_id]
                ]
            )

            # Calculate additional metrics
            avg_tests_per_fault = (
                num_fault_revealing_tests / num_faults if num_faults > 0 else 0
            )
            pct_tests_revealing = (
                (num_fault_revealing_tests / num_tests * 100) if num_tests > 0 else 0
            )

            data.append(
                {
                    "Test Suite": f"{program}_{version}",
                    "Program": program,
                    "Version": version,
                    "Dataset": "D4J" if (program, version) in D4J else "SIR",
                    "Number of Tests": num_tests,
                    "Number of Faults": num_faults,
                    "Tests Revealing Faults": num_fault_revealing_tests,
                    "Avg Tests per Fault": round(avg_tests_per_fault, 2),
                    "Pct Tests Revealing Faults": round(pct_tests_revealing, 1),
                    "Avg Char Length": round(avg_char_len, 1),
                    "Avg Word Count": round(avg_word_count, 1),
                    "Avg Token Count": round(avg_token_count, 1),
                    "Min Token Count": min_token_count,
                    "Max Token Count": max_token_count,
                }
            )

        except Exception as e:
            print(f"Error loading {program}_{version}: {e}")
            continue

    return pd.DataFrame(data)


def display_table(df):
    """Display the dataframe as a rich table."""
    console = Console()

    table = Table(title="Testset Information")

    # Add columns
    for col in df.columns:
        table.add_column(col, justify="center")

    # Add rows
    for _, row in df.iterrows():
        table.add_row(*[str(val) for val in row])

    # Add separator row
    table.add_row(*["─" * 10 for _ in df.columns], style="dim")

    # Add total row (bold)
    total_tests = df["Number of Tests"].sum()
    total_faults = df["Number of Faults"].sum()
    total_revealing = df["Tests Revealing Faults"].sum()
    avg_tests_fault = (
        round(total_revealing / total_faults, 2) if total_faults > 0 else 0
    )
    pct_revealing = (
        round((total_revealing / total_tests * 100), 1) if total_tests > 0 else 0
    )

    table.add_row(
        "[bold]TOTAL[/bold]",
        "[bold]--[/bold]",
        "[bold]--[/bold]",
        "[bold]--[/bold]",
        f"[bold]{total_tests}[/bold]",
        f"[bold]{total_faults}[/bold]",
        f"[bold]{total_revealing}[/bold]",
        f"[bold]{avg_tests_fault}[/bold]",
        f"[bold]{pct_revealing}[/bold]",
    )

    console.print(table)

    # Also print summary statistics
    console.print("\n[bold]Summary Statistics:[/bold]")
    console.print(f"Total test suites: {len(df)}")
    console.print(f"Total tests: {df['Number of Tests'].sum()}")
    console.print(f"Total faults: {df['Number of Faults'].sum()}")
    console.print(f"Total fault-revealing tests: {df['Tests Revealing Faults'].sum()}")


def generate_latex_table(df):
    """Generate LaTeX table format with fixed column widths."""
    # Remove Test Suite column and adjust data
    df_latex = df[
        [
            "Program",
            "Version",
            "Dataset",
            "Number of Tests",
            "Number of Faults",
            "Tests Revealing Faults",
            "Avg Tests per Fault",
            "Pct Tests Revealing Faults",
            "Avg Char Length",
            "Avg Word Count",
            "Avg Token Count",
            "Min Token Count",
            "Max Token Count",
        ]
    ].copy()

    # Define column formats with fixed width (1.2cm each for 13 columns = ~15.6cm total width)
    col_width = "1.2cm"
    col_format = f"p{{{col_width}}}" * len(df_latex.columns)

    # Headers with bold formatting
    headers = [
        "\\textbf{Program}",
        "\\textbf{Version}",
        "\\textbf{Dataset}",
        "\\textbf{\\#T}",
        "\\textbf{\\#F}",
        "\\textbf{TRF}",
        "\\textbf{T/F}",
        "\\textbf{\\%TRF}",
        "\\textbf{Avg Char Len}",
        "\\textbf{Avg Word Cnt}",
        "\\textbf{Avg Tok Cnt}",
        "\\textbf{Min Tok Cnt}",
        "\\textbf{Max Tok Cnt}",
    ]

    # Generate LaTeX table
    latex_table = []

    # Table header (two-column float, top placement)
    latex_table.append("\\begin{table*}[t]")
    latex_table.append("\\centering")
    latex_table.append("\\begin{tabular}{" + col_format + "}")
    latex_table.append("\\hline")

    # Column headers
    header_row = " & ".join(headers)
    latex_table.append(f"{header_row} \\\\")
    latex_table.append("\\hline")

    # Data rows
    for _, row in df_latex.iterrows():
        row_str = " & ".join(str(val) for val in row)
        latex_table.append(f"{row_str} \\\\")

    # Add horizontal rule before total row
    latex_table.append("\\hline")

    # Add total row (bold)
    total_tests = df_latex["Number of Tests"].sum()
    total_faults = df_latex["Number of Faults"].sum()
    total_revealing = df_latex["Tests Revealing Faults"].sum()
    avg_tests_fault = (
        round(total_revealing / total_faults, 2) if total_faults > 0 else 0
    )
    pct_revealing = (
        round((total_revealing / total_tests * 100), 1) if total_tests > 0 else 0
    )

    total_row = f"\\textbf{{TOTAL}} & \\textbf{{--}} & \\textbf{{--}} & \\textbf{{--}} & \\textbf{{{total_tests}}} & \\textbf{{{total_faults}}} & \\textbf{{{total_revealing}}} & \\textbf{{{avg_tests_fault}}} & \\textbf{{{pct_revealing}}} & \\textbf{{--}} & \\textbf{{--}} & \\textbf{{--}} & \\textbf{{--}} & \\textbf{{--}}"
    latex_table.append(f"{total_row} \\\\")

    # Table footer
    latex_table.append("\\hline")
    latex_table.append("\\end{tabular}")
    latex_table.append("\\caption{Test suite Information\\\\")
    latex_table.append(
        "\\tiny{\\#T: Number of Tests, \\#F: Number of Faults, TRF: Tests Revealing Faults,\\\\"
    )
    latex_table.append(
        "T/F: Average Tests per Fault, \\%TRF: Percentage Tests Revealing Faults,\\\\"
    )
    latex_table.append(
        "Avg Char Len: Average Character Length, Avg Word Cnt: Average Word Count,\\\\"
    )
    latex_table.append(
        "Avg Tok Cnt: Average Token Count, Min/Max Tok Cnt: Min/Max Token Count}}"
    )
    latex_table.append("\\footnotesize")
    latex_table.append("\\label{tab:testset_info}")
    latex_table.append("\\end{table*}")

    return "\n".join(latex_table)


def main():
    """Main function."""
    df = get_testset_info()

    if df.empty:
        print("No test suites could be loaded.")
        return

    # Sort by dataset and program
    df = df.sort_values(["Dataset", "Program"])

    # Display the table
    display_table(df)

    # Save to CSV
    csv_file = "testset_info.csv"
    df.to_csv(csv_file, index=False)
    print(f"\nCSV table saved to {csv_file}")

    # Generate and save LaTeX table
    latex_table = generate_latex_table(df)
    latex_file = "testset_info.tex"
    with open(latex_file, "w") as f:
        f.write(latex_table)
    print(f"LaTeX table saved to {latex_file}")


if __name__ == "__main__":
    main()
