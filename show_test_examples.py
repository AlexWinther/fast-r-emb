#!/usr/bin/env python3
"""
Script to show example test cases from both C-based (SIR) and Java-based (D4J) test suites.
Includes LaTeX formatting for report appendices.
"""

import sys
from pathlib import Path

# Add src to path to import local modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils import load_test_suite
from rich.console import Console
from rich.panel import Panel


def format_java_for_latex(java_code, max_lines=50):
    """Format Java code for LaTeX inclusion with syntax highlighting."""
    # Ensure we have proper line breaks - sometimes the code might be stored differently
    if "\n" not in java_code and len(java_code) > 200:
        # If it's one long line, try to add reasonable line breaks
        java_code = java_code.replace(";", ";\n")
        java_code = java_code.replace("{", "{\n")
        java_code = java_code.replace("}", "\n}\n")
        java_code = java_code.replace("*/", "*/\n")

    lines = java_code.split("\n")

    # Clean up and reformat lines for better readability
    formatted_lines = []
    indent_level = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Adjust indentation based on braces
        if "}" in line and not line.startswith("}"):
            indent_level = max(0, indent_level - 1)

        # Add proper indentation
        indented_line = "    " * indent_level + line

        # Adjust indent level after opening braces
        if "{" in line and not line.endswith("{"):
            indent_level += 1

        formatted_lines.append(indented_line)

    # Limit to max_lines to keep it reasonable for appendix
    if len(formatted_lines) > max_lines:
        formatted_lines = formatted_lines[: max_lines - 1] + [
            f"// ... ({len(formatted_lines) - max_lines + 1} more lines truncated for brevity) ..."
        ]

    # Basic LaTeX escaping
    escaped_lines = []
    for line in formatted_lines:
        # Escape LaTeX special characters
        line = line.replace("\\", "\\textbackslash{}")
        line = line.replace("&", "\\&")
        line = line.replace("%", "\\%")
        line = line.replace("$", "\\$")
        line = line.replace("#", "\\#")
        line = line.replace("_", "\\_")
        line = line.replace("{", "\\{")
        line = line.replace("}", "\\}")
        line = line.replace("~", "\\textasciitilde{}")
        line = line.replace("^", "\\textasciicircum{}")
        escaped_lines.append(line)

    return "\n".join(escaped_lines)


def generate_c_test_listing(c_test_cases, test_suite_name):
    """Generate a small LaTeX listing for C test cases suitable for main report text."""
    # Show first 3 test cases
    examples = c_test_cases[:3]

    latex_listing = f"""\\begin{{lstlisting}}[language=bash, basicstyle=\\ttfamily\\footnotesize, caption=Example C test cases from {test_suite_name}, label=lst:{test_suite_name.lower().replace("_", "")}_c_tests]
# Test case 1:
grep {examples[0]}

# Test case 2:
grep {examples[1]}

# Test case 3:
grep {examples[2]}
\\end{{lstlisting}}"""

    return latex_listing


def generate_latex_appendix(java_code, class_name, test_suite_name):
    """Generate LaTeX code for appendix inclusion."""
    formatted_code = format_java_for_latex(java_code)

    latex_appendix = f"""% LaTeX Appendix: {test_suite_name} Test Case Example
% Required packages: \\usepackage{{listings}} \\usepackage{{xcolor}}
% For syntax highlighting: \\lstset{{language=Java, basicstyle=\\ttfamily\\footnotesize, keywordstyle=\\color{{blue}}, commentstyle=\\color{{green!60!black}}, stringstyle=\\color{{red}}}}

\\newpage
\\section{{{test_suite_name} Test Case Example}}
\\label{{app:{test_suite_name.lower().replace("_", "")}_example}}

This appendix shows an example test case from the {test_suite_name} test suite.

\\begin{{lstlisting}}[language=Java, caption=Example JUnit test class from {test_suite_name}, label=lst:{test_suite_name.lower().replace("_", "")}_test]
{formatted_code}
\\end{{lstlisting}}

This test case demonstrates the structure of JUnit tests used in the {test_suite_name} dataset, including test methods, assertions, and setup code.
"""

    return latex_appendix


def show_test_examples():
    """Show example test cases from both C and Java test suites."""
    console = Console()

    console.print(
        "\n[bold blue]Test Case Examples from Different Test Suites[/bold blue]\n"
    )

    # Example from C-based test suite (SIR)
    c_program, c_version = "grep", "v3"  # Using grep as example
    console.print(
        f"[bold green]C-based Test Suite Example: {c_program}_{c_version}[/bold green]"
    )

    try:
        c_test_suite = load_test_suite(c_program, c_version)

        # Show first few test cases
        c_examples = c_test_suite.test_cases[:3]  # Show first 3 examples

        for i, test_case in enumerate(c_examples, 1):
            # Format as command line
            panel = Panel.fit(
                f"[cyan]$ grep {test_case}[/cyan]",
                title=f"Test Case {i}",
                border_style="blue",
            )
            console.print(panel)

    except Exception as e:
        console.print(f"[red]Error loading C test suite: {e}[/red]")
    else:
        # Generate LaTeX listing for C tests
        latex_c_listing = generate_c_test_listing(
            c_test_suite.test_cases, f"{c_program}_{c_version}"
        )
        c_latex_file = f"c_tests_{c_program}_{c_version}_listing.tex"
        with open(c_latex_file, "w") as f:
            f.write(latex_c_listing)
        console.print(f"\n[dim]LaTeX C test listing saved to {c_latex_file}[/dim]")

    console.print()

    # Example from Java-based test suite (D4J)
    java_program, java_version = "chart", "v1"  # Using chart as example
    console.print(
        f"[bold green]Java-based Test Suite Example: {java_program}_{java_version}[/bold green]"
    )

    try:
        java_test_suite = load_test_suite(java_program, java_version)

        # Show first test case (Java files are much longer, so show just the beginning)
        if java_test_suite.test_cases:
            java_example = java_test_suite.test_cases[0]

            # Show first 20 lines of the Java test file
            lines = java_example.split("\n")[:20]
            preview = "\n".join(lines)

            # Try to extract the class name from the file
            class_name = "Unknown"
            for line in lines:
                if "class" in line and "Tests.java" in line:
                    # Extract class name from something like "* CategoryLineAnnotationTests.java *"
                    parts = line.split()
                    if len(parts) > 1:
                        class_name = parts[1].replace("*", "").replace(".java", "")
                        break

            panel = Panel.fit(
                f"[yellow]{preview}[/yellow]\n[gray]... (file continues)[/gray]",
                title=f"Test Class: {class_name}",
                border_style="green",
            )
            console.print(panel)

            # Generate LaTeX appendix format
            latex_appendix = generate_latex_appendix(
                java_example, class_name, f"{java_program}_{java_version}"
            )
            latex_file = f"appendix_{java_program}_{java_version}_example.tex"
            with open(latex_file, "w") as f:
                f.write(latex_appendix)
            console.print(f"\n[dim]LaTeX appendix saved to {latex_file}[/dim]")

    except Exception as e:
        console.print(f"[red]Error loading Java test suite: {e}[/red]")

    console.print("\n[bold]Summary:[/bold]")
    console.print(
        "• C-based test suites (SIR): Test cases are command-line arguments for testing programs"
    )
    console.print(
        "• Java-based test suites (D4J): Test cases are complete JUnit test class files"
    )
    console.print("• Both types help detect faults in their respective programs")
    console.print(
        "\n[dim]Note: LaTeX appendix format generated for Java test case examples[/dim]"
    )


def main():
    """Main function."""
    show_test_examples()


if __name__ == "__main__":
    main()
