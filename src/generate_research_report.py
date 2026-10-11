
from pathlib import Path
from datetime import datetime
import json
import re
import html

import pandas as pd

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    LongTable,
    TableStyle,
    Image,
    PageBreak,
    KeepTogether,
)

# =========================================================
# PATHS
# =========================================================

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "outputs" / "results"
FIGURES = ROOT / "outputs" / "figures"
REPORT_DIR = ROOT / "report"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

PDF_PATH = REPORT_DIR / "credit_risk_research_report_revised.pdf"

PAGE_WIDTH, PAGE_HEIGHT = A4
LEFT_MARGIN = 0.65 * inch
RIGHT_MARGIN = 0.65 * inch
AVAILABLE_WIDTH = PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN

NAVY = colors.HexColor("#183153")
LIGHT_GREY = colors.HexColor("#F1F5F9")
GRID_GREY = colors.HexColor("#D1D5DB")
DARK_GREY = colors.HexColor("#444444")

story = []

# =========================================================
# STYLES
# =========================================================

styles = getSampleStyleSheet()

styles.add(
    ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=23,
        leading=29,
        textColor=NAVY,
        spaceAfter=18,
    )
)

styles.add(
    ParagraphStyle(
        name="CoverSubtitle",
        parent=styles["Heading2"],
        alignment=TA_CENTER,
        fontSize=12,
        leading=17,
        textColor=DARK_GREY,
        spaceAfter=12,
    )
)

styles.add(
    ParagraphStyle(
        name="SectionHeading",
        parent=styles["Heading1"],
        fontSize=15,
        leading=19,
        textColor=NAVY,
        spaceBefore=12,
        spaceAfter=8,
        keepWithNext=True,
    )
)

styles.add(
    ParagraphStyle(
        name="SubHeading",
        parent=styles["Heading2"],
        fontSize=10,
        leading=13,
        textColor=NAVY,
        spaceBefore=8,
        spaceAfter=5,
        keepWithNext=True,
    )
)

styles.add(
    ParagraphStyle(
        name="ReportBody",
        parent=styles["BodyText"],
        fontSize=9,
        leading=13,
        spaceAfter=7,
    )
)

styles.add(
    ParagraphStyle(
        name="SmallNote",
        parent=styles["BodyText"],
        fontSize=7.5,
        leading=10,
        textColor=DARK_GREY,
        spaceAfter=5,
    )
)

styles.add(
    ParagraphStyle(
        name="TableHeader",
        parent=styles["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=6.8,
        leading=8,
        textColor=colors.white,
        alignment=TA_CENTER,
    )
)

styles.add(
    ParagraphStyle(
        name="TableCell",
        parent=styles["BodyText"],
        fontSize=7,
        leading=8.5,
        wordWrap="CJK",
    )
)

styles.add(
    ParagraphStyle(
        name="TableCellSmall",
        parent=styles["BodyText"],
        fontSize=5.8,
        leading=7,
        wordWrap="CJK",
    )
)

# =========================================================
# GENERAL HELPERS
# =========================================================

def add_paragraph(text, style="ReportBody"):
    """Add a paragraph to the PDF."""
    story.append(Paragraph(text, styles[style]))


def add_heading(text):
    add_paragraph(text, "SectionHeading")


def load_csv(filename):
    """Load a results CSV if it exists."""
    path = RESULTS / filename

    if not path.exists():
        return None

    try:
        return pd.read_csv(path)
    except (pd.errors.ParserError, UnicodeDecodeError, OSError) as exc:
        print(f"Warning: Could not read {filename}: {exc}")
        return None


def load_json(filename):
    """Load a JSON file if it exists."""
    path = RESULTS / filename

    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"Warning: Could not read {filename}: {exc}")
        return None


def nested_get(data, key):
    """Read a value using dot notation for nested JSON keys."""
    value = data

    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None

        value = value[part]

    return value


def format_value(value):
    """Format numeric and missing values for tables."""
    if pd.isna(value):
        return "—"

    if isinstance(value, (float,)) or hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, AttributeError):
            pass

    if isinstance(value, float):
        return f"{value:.4f}"

    return str(value)


def safe_cell(value, small=False):
    """Create a wrapped, XML-safe ReportLab table cell."""
    text = html.escape(format_value(value))
    style_name = "TableCellSmall" if small else "TableCell"

    return Paragraph(text, styles[style_name])


def add_missing_note(title, filename):
    add_paragraph(
        f"{html.escape(title)} could not be displayed because "
        f"<i>{html.escape(filename)}</i> was not found or could not "
        f"be read in outputs/results.",
        "SmallNote",
    )


def add_table(
    title,
    filename,
    columns,
    max_rows=15,
    column_widths=None,
    small=False,
    header_labels=None,
):
    """Create a wrapped, consistently styled table from a results CSV."""
    df = load_csv(filename)

    if df is None or df.empty:
        add_missing_note(title, filename)
        return

    selected = [column for column in columns if column in df.columns]

    if not selected:
        print(
            f"Warning: None of the requested columns were found "
            f"in {filename}."
        )
        add_missing_note(title, filename)
        return

    df = df[selected].head(max_rows)

    labels = header_labels or {}

    headers = [
        labels.get(
            column,
            str(column).replace("_", " ").title(),
        )
        for column in selected
    ]

    data = [
        [Paragraph(html.escape(str(header)), styles["TableHeader"])
         for header in headers]
    ]

    for row in df.itertuples(index=False, name=None):
        data.append([
            safe_cell(value, small=small)
            for value in row
        ])

    if column_widths is None:
        column_widths = [
            AVAILABLE_WIDTH / len(selected)
        ] * len(selected)

    if len(column_widths) != len(selected):
        raise ValueError(
            f"Table width configuration for {filename} does not "
            "match the number of selected columns."
        )

    table_class = LongTable if len(data) > 12 else Table

    table = table_class(
        data,
        colWidths=column_widths,
        repeatRows=1,
        hAlign="LEFT",
    )

    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.35, GRID_GREY),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.white, LIGHT_GREY],
                ),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )

    story.append(Paragraph(title, styles["SubHeading"]))
    story.append(table)
    story.append(Spacer(1, 9))


def add_figure(filename, caption):
    """Add a figure and caption, or a note if it is missing."""
    path = FIGURES / filename

    if not path.exists():
        print(f"Warning: Figure not found: {path}")
        add_paragraph(
            f"Figure unavailable: {html.escape(filename)}",
            "SmallNote",
        )
        return

    try:
        img = Image(str(path))
        img._restrictSize(
            AVAILABLE_WIDTH,
            3.65 * inch,
        )

        story.append(
            KeepTogether(
                [
                    img,
                    Spacer(1, 4),
                    Paragraph(
                        html.escape(caption),
                        styles["SmallNote"],
                    ),
                    Spacer(1, 10),
                ]
            )
        )
    except (OSError, ValueError) as exc:
        print(f"Warning: Could not add figure {filename}: {exc}")
        add_paragraph(
            f"Figure could not be loaded: {html.escape(filename)}",
            "SmallNote",
        )


def add_page_number(canvas, doc):
    """Add a footer and page number to every page."""
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(DARK_GREY)

    canvas.drawString(
        LEFT_MARGIN,
        0.4 * inch,
        "Credit Risk Research Capstone",
    )

    canvas.drawRightString(
        PAGE_WIDTH - RIGHT_MARGIN,
        0.4 * inch,
        f"Page {doc.page}",
    )

    canvas.restoreState()


# =========================================================
# DATASET VALIDATION SUMMARY
# =========================================================

def add_json_summary(filename):
    """Display key dataset-validation results without dumping all columns."""
    report = load_json(filename)

    if not isinstance(report, dict):
        add_missing_note("Dataset validation summary", filename)
        return

    summary_fields = [
        ("dataset_name", "Dataset"),
        ("source", "Source"),
        ("raw_file", "Raw data file"),
        ("standardized_file", "Processed data file"),
        ("rows", "Rows"),
        ("columns", "Columns"),
        ("total_missing_values", "Missing values"),
        ("duplicate_rows", "Exact duplicate rows"),
        ("duplicate_rows_removed", "Duplicate rows removed"),
        ("default_rate_percent", "Observed default rate (%)"),
        ("all_checks_passed", "All validation checks passed"),
    ]

    rows = [
        [
            Paragraph("Validation check", styles["TableHeader"]),
            Paragraph("Result", styles["TableHeader"]),
        ]
    ]

    for key, label in summary_fields:
        value = nested_get(report, key)

        if value is None:
            continue

        if key == "default_rate_percent":
            try:
                value = f"{float(value):.2f}%"
            except (ValueError, TypeError):
                pass

        rows.append(
            [
                safe_cell(label),
                safe_cell(value),
            ]
        )

    if len(rows) == 1:
        add_paragraph(
            "The validation JSON was read, but no recognised "
            "summary fields were found.",
            "SmallNote",
        )
        return

    table = Table(
        rows,
        colWidths=[
            AVAILABLE_WIDTH * 0.38,
            AVAILABLE_WIDTH * 0.62,
        ],
        repeatRows=1,
    )

    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("GRID", (0, 0), (-1, -1), 0.35, GRID_GREY),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.white, LIGHT_GREY],
                ),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )

    story.append(
        Paragraph("Dataset validation summary", styles["SubHeading"])
    )
    story.append(table)
    story.append(Spacer(1, 10))


# =========================================================
# SHAP IMPORTANCE TABLE
# =========================================================

def add_shap_importance_table():
    """
    Locate a likely saved SHAP-importance CSV and display both feature
    names and importance scores when the required columns are present.
    """
    candidates = [
        "shap_feature_importance.csv",
        "shap_importance.csv",
        "feature_importance.csv",
        "shap_values_summary.csv",
    ]

    df = None
    selected_file = None

    for filename in candidates:
        candidate_df = load_csv(filename)

        if candidate_df is not None and not candidate_df.empty:
            df = candidate_df
            selected_file = filename
            break

    if df is None:
        # Search only filenames clearly related to SHAP/importance.
        for path in sorted(RESULTS.glob("*.csv")):
            if not re.search(
                r"(shap|feature.importance|importance)",
                path.stem,
                flags=re.IGNORECASE,
            ):
                continue

            candidate_df = load_csv(path.name)

            if candidate_df is not None and not candidate_df.empty:
                df = candidate_df
                selected_file = path.name
                break

    if df is None:
        add_paragraph(
            "The SHAP figures are included below. A separate importance "
            "table was not added because no suitable feature-importance "
            "CSV was found in outputs/results. No importance values "
            "have been estimated or invented.",
            "SmallNote",
        )
        return

    # Detect a feature-name column.
    feature_candidates = [
        "feature",
        "feature_name",
        "features",
        "variable",
        "name",
    ]

    feature_col = next(
        (
            column for column in df.columns
            if column.lower() in feature_candidates
        ),
        None,
    )

    if feature_col is None:
        feature_col = next(
            (
                column for column in df.columns
                if "feature" in column.lower()
                or "variable" in column.lower()
            ),
            None,
        )

    # Detect an importance-value column.
    importance_col = None

    importance_candidates = [
        "mean_abs_shap",
        "mean_absolute_shap",
        "mean_abs_shap_value",
        "mean_absolute_shap_value",
        "mean_shap",
        "importance",
        "importance_score",
        "mean_abs_value",
    ]

    for candidate in importance_candidates:
        importance_col = next(
            (
                column for column in df.columns
                if column.lower() == candidate
            ),
            None,
        )

        if importance_col is not None:
            break

    if importance_col is None:
        importance_col = next(
            (
                column for column in df.columns
                if "importance" in column.lower()
                or (
                    "shap" in column.lower()
                    and any(
                        term in column.lower()
                        for term in ("mean", "abs", "value")
                    )
                )
            ),
            None,
        )

    if feature_col is None or importance_col is None:
        add_paragraph(
            f"The file {html.escape(selected_file)} was found, but its "
            "columns did not clearly identify both feature names and "
            "SHAP importance scores. The values have not been guessed.",
            "SmallNote",
        )
        return

    output = df[[feature_col, importance_col]].copy()
    output[importance_col] = pd.to_numeric(
        output[importance_col],
        errors="coerce",
    )
    output = output.dropna(subset=[importance_col])

    if output.empty:
        add_paragraph(
            "No numeric SHAP importance values were found in the "
            "detected file.",
            "SmallNote",
        )
        return

    output = output.sort_values(
        importance_col,
        ascending=False,
    ).head(15)

    output.columns = ["Feature", "Mean absolute SHAP"]

    # Reuse the standard table styling while keeping this function
    # independent of the source CSV's original column names.
    data = [
        [
            Paragraph("Feature", styles["TableHeader"]),
            Paragraph("Mean absolute SHAP", styles["TableHeader"]),
        ]
    ]

    for row in output.itertuples(index=False, name=None):
        data.append(
            [
                safe_cell(row[0]),
                safe_cell(row[1]),
            ]
        )

    table = Table(
        data,
        colWidths=[
            AVAILABLE_WIDTH * 0.66,
            AVAILABLE_WIDTH * 0.34,
        ],
        repeatRows=1,
    )

    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("GRID", (0, 0), (-1, -1), 0.35, GRID_GREY),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.white, LIGHT_GREY],
                ),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )

    story.append(
        Paragraph(
            "Leading SHAP feature-importance results",
            styles["SubHeading"],
        )
    )
    story.append(table)
    story.append(Spacer(1, 10))


# =========================================================
# COVER PAGE
# =========================================================

story.append(Spacer(1, 1.15 * inch))

add_paragraph(
    "CREDIT RISK RESEARCH CAPSTONE",
    "ReportTitle",
)

add_paragraph(
    "Predictive Modelling, Calibration, Explainability "
    "and Fairness Assessment",
    "CoverSubtitle",
)

story.append(Spacer(1, 0.35 * inch))

add_paragraph(
    "<b>Dataset:</b> UCI Default of Credit Card Clients dataset "
    "(historical Taiwan credit-card data)."
)

add_paragraph(
    "<b>Objective:</b> Evaluate machine-learning approaches for "
    "identifying customers who may default on future credit-card "
    "payments, considering predictive performance, probability "
    "quality, threshold selection and subgroup evaluation."
)

add_paragraph(
    f"<b>Report generated:</b> {datetime.now().strftime('%d %B %Y')}"
)

story.append(Spacer(1, 0.4 * inch))

add_paragraph(
    "This report summarizes the analyses and saved results available "
    "in the project at generation time. It is an educational research "
    "exercise, not a validated lending system or a recommendation for "
    "individual credit decisions.",
    "SmallNote",
)

story.append(PageBreak())

# =========================================================
# 1. EXECUTIVE SUMMARY
# =========================================================

add_heading("1. Executive Summary")

add_paragraph(
    "This project investigates credit-card default prediction using "
    "logistic regression, random forest and gradient boosting. The "
    "workflow includes data validation, exploratory analysis, model "
    "comparison, calibration assessment, SHAP-based explanation, "
    "threshold analysis and subgroup evaluation."
)

add_paragraph(
    "In the saved final evaluation, gradient boosting achieved a "
    "ROC-AUC of approximately 0.7849 and average precision of "
    "approximately 0.5517. Its Brier score was approximately 0.1342. "
    "These results describe this dataset and evaluation procedure; "
    "they do not establish performance on future customers or other "
    "populations."
)

add_paragraph(
    "The operating threshold affects the trade-off between detecting "
    "defaults and producing false-positive predictions. Subgroup "
    "estimates also differ and have varying uncertainty. These findings "
    "support further validation rather than a conclusion that the model "
    "is fair or ready for operational use."
)

# =========================================================
# 2. DATASET AND VALIDATION
# =========================================================

add_heading("2. Dataset and Data Validation")

add_paragraph(
    "The working dataset contains 30,000 rows and 24 columns, "
    "including 23 input features and the target variable "
    "<i>default_payment_next_month</i>. The observed default rate is "
    "22.12%. The project validation reported no missing values and "
    "35 exact duplicate rows."
)

add_paragraph(
    "The working CSV does not include the original customer ID. "
    "Consequently, identical rows cannot automatically be assumed "
    "to represent the same individual, and duplicate rows were "
    "retained. The historical Taiwan context should be considered "
    "when interpreting these results."
)

add_json_summary("dataset_validation_report.json")

add_figure(
    "target_distribution.png",
    "Figure 1. Distribution of the target classes.",
)

add_figure(
    "age_by_default.png",
    "Figure 2. Age distribution by observed default outcome.",
)

add_figure(
    "credit_limit_by_default.png",
    "Figure 3. Credit-limit distribution by observed default outcome.",
)

add_figure(
    "default_rate_by_pay0.png",
    "Figure 4. Observed default rate by PAY_0 repayment-status category.",
)

# =========================================================
# 3. METHODOLOGY
# =========================================================

add_heading("3. Methodology")

add_paragraph(
    "<b>Data preparation:</b> The project checks column names, "
    "missing values, duplicate rows, target values and basic "
    "data distributions."
)

add_paragraph(
    "<b>Model development:</b> Logistic regression, random forest "
    "and gradient boosting are compared. Categorical variables are "
    "encoded and numerical variables are scaled in the logistic-"
    "regression pipeline."
)

add_paragraph(
    "<b>Evaluation design:</b> The final evaluation script uses a "
    "stratified 60% training, 20% validation and 20% test split, "
    "with random seed 2026. Model selection is based on validation "
    "ROC-AUC. The selected model is then retrained on training plus "
    "validation data before final test evaluation."
)

add_paragraph(
    "<b>Metrics:</b> ROC-AUC, average precision, accuracy, precision, "
    "recall, F1, Brier score and log loss evaluate different aspects "
    "of model performance. No single metric fully describes the "
    "practical suitability of a credit-risk model."
)

add_paragraph(
    "<b>Important qualification:</b> Earlier exploratory modelling "
    "used the same dataset before the final split was introduced. "
    "The final test set is therefore not an entirely untouched "
    "external evaluation. Independent external validation would "
    "provide stronger evidence of generalization."
)

# =========================================================
# 4. FINAL MODEL EVALUATION
# =========================================================

add_heading("4. Final Model Evaluation")

add_paragraph(
    "The following table summarizes the saved final test results. "
    "Higher ROC-AUC and average precision indicate better ranking "
    "performance, while lower Brier score and log loss indicate "
    "better probabilistic predictions."
)

add_table(
    "Table 1. Final test-set model comparison",
    "final_test_metrics.csv",
    [
        "model",
        "roc_auc",
        "average_precision",
        "brier_score",
        "log_loss",
    ],
    max_rows=10,
    column_widths=[
        AVAILABLE_WIDTH * 0.28,
        AVAILABLE_WIDTH * 0.16,
        AVAILABLE_WIDTH * 0.22,
        AVAILABLE_WIDTH * 0.16,
        AVAILABLE_WIDTH * 0.18,
    ],
    header_labels={
        "model": "Model",
        "roc_auc": "ROC-AUC",
        "average_precision": "Avg. precision",
        "brier_score": "Brier score",
        "log_loss": "Log loss",
    },
)

add_figure(
    "final_test_calibration.png",
    "Figure 5. Calibration of predicted probabilities on the final test set.",
)

# =========================================================
# 5. MODEL EXPLAINABILITY
# =========================================================

add_heading("5. Model Explainability")

add_paragraph(
    "SHAP was used to examine which features influenced the "
    "gradient-boosting model's predictions. The saved analysis ranks "
    "PAY_0 among the most influential features, followed by LIMIT_BAL, "
    "PAY_3 and several payment-amount variables."
)

add_paragraph(
    "Feature importance describes model behaviour and associations "
    "in the analysed data. It does not establish causality, prove "
    "that a feature is appropriate for lending, or demonstrate that "
    "the model is free from unfair effects."
)

add_shap_importance_table()

add_figure(
    "shap_feature_importance.png",
    "Figure 6. Mean absolute SHAP feature importance.",
)

add_figure(
    "shap_summary.png",
    "Figure 7. SHAP summary of feature contributions.",
)

# =========================================================
# 6. DECISION THRESHOLD ANALYSIS
# =========================================================

add_heading("6. Decision Threshold Analysis")

add_paragraph(
    "A probability threshold determines when a prediction is labelled "
    "as a likely default. The threshold analysis selected 0.25 based "
    "on validation-set F1 among the tested thresholds. This is a "
    "modelling choice, not a universal lending policy."
)

add_table(
    "Table 2. Threshold comparison on the final test set",
    "final_threshold_comparison.csv",
    [
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "tn",
        "fp",
        "fn",
        "tp",
    ],
    max_rows=10,
    small=True,
    header_labels={
        "threshold": "Threshold",
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
        "tn": "TN",
        "fp": "FP",
        "fn": "FN",
        "tp": "TP",
    },
)

add_figure(
    "final_threshold_comparison.png",
    "Figure 8. Performance comparison between the evaluated thresholds.",
)

add_paragraph(
    "In the saved final test results, threshold 0.25 achieved recall "
    "of approximately 0.5923 and F1 of approximately 0.5574. Threshold "
    "0.50 achieved recall of approximately 0.3632 and F1 of "
    "approximately 0.4735. The lower threshold identified more actual "
    "defaults but also generated more false positives."
)

add_paragraph(
    "The saved results indicate that threshold 0.25 produced 786 true "
    "positives and 707 false positives, compared with 482 true "
    "positives and 227 false positives at threshold 0.50. The "
    "appropriate trade-off depends on the costs and consequences "
    "of both types of error."
)

# =========================================================
# 7. SUBGROUP EVALUATION AND UNCERTAINTY
# =========================================================

add_heading("7. Subgroup Evaluation and Uncertainty")

add_paragraph(
    "The project compares recall and false-positive rates across "
    "sex-coded categories and age groups. Wilson 95% confidence "
    "intervals show sampling uncertainty for subgroup recall and "
    "false-positive-rate estimates."
)

add_table(
    "Table 3. Subgroup recall and false-positive-rate confidence intervals",
    "fairness_uncertainty_intervals.csv",
    [
        "grouping",
        "group",
        "sample_size",
        "recall",
        "recall_ci_lower",
        "recall_ci_upper",
        "false_positive_rate",
        "fpr_ci_lower",
        "fpr_ci_upper",
    ],
    max_rows=15,
    small=True,
    column_widths=[
        AVAILABLE_WIDTH * 0.10,
        AVAILABLE_WIDTH * 0.12,
        AVAILABLE_WIDTH * 0.11,
        AVAILABLE_WIDTH * 0.10,
        AVAILABLE_WIDTH * 0.11,
        AVAILABLE_WIDTH * 0.11,
        AVAILABLE_WIDTH * 0.13,
        AVAILABLE_WIDTH * 0.11,
        AVAILABLE_WIDTH * 0.11,
    ],
    header_labels={
        "grouping": "Group type",
        "group": "Group",
        "sample_size": "N",
        "recall": "Recall",
        "recall_ci_lower": "Recall CI low",
        "recall_ci_upper": "Recall CI high",
        "false_positive_rate": "FPR",
        "fpr_ci_lower": "FPR CI low",
        "fpr_ci_upper": "FPR CI high",
    },
)

add_figure(
    "fairness_recall_ci_by_sex.png",
    "Figure 9. Recall estimates and confidence intervals by sex-coded group.",
)

add_figure(
    "fairness_recall_ci_by_age.png",
    "Figure 10. Recall estimates and confidence intervals by age group.",
)

add_figure(
    "fairness_fpr_ci_by_sex.png",
    "Figure 11. False-positive-rate estimates and confidence intervals by sex-coded group.",
)

add_figure(
    "fairness_fpr_ci_by_age.png",
    "Figure 12. False-positive-rate estimates and confidence intervals by age group.",
)

add_paragraph(
    "Subgroup differences warrant investigation but do not by "
    "themselves prove discrimination or establish fairness. The "
    "60-and-over test subgroup contains only 60 records, producing "
    "particularly wide confidence intervals. The dataset's coded "
    "categories should be verified against the original documentation "
    "before substantive interpretation."
)

# =========================================================
# 8. LIMITATIONS AND RESPONSIBLE USE
# =========================================================

add_heading("8. Limitations and Responsible Use")

add_paragraph(
    "<b>1. Population and time:</b> The dataset is historical and "
    "specific to Taiwan. Results should not be assumed to generalize "
    "to other countries, institutions or time periods."
)

add_paragraph(
    "<b>2. Evaluation independence:</b> Earlier exploratory work used "
    "the same dataset. The final split does not replace independent "
    "external validation."
)

add_paragraph(
    "<b>3. Threshold choice:</b> A suitable threshold depends on the "
    "relative costs of missed defaults and false alarms, operational "
    "requirements and applicable rules."
)

add_paragraph(
    "<b>4. Fairness:</b> Confidence intervals describe uncertainty in "
    "subgroup rates. They are not a complete fairness assessment."
)

add_paragraph(
    "<b>5. Explainability:</b> SHAP describes model behaviour and "
    "should not be interpreted as causal evidence."
)

add_paragraph(
    "<b>6. Research-only status:</b> This project is educational "
    "research. It is not a validated or approved system for making "
    "real credit decisions."
)

# =========================================================
# 9. CONCLUSION
# =========================================================

add_heading("9. Conclusion")

add_paragraph(
    "The project demonstrates an end-to-end credit-risk research "
    "workflow covering data validation, exploratory analysis, model "
    "comparison, probability calibration, feature explanation, "
    "threshold analysis and subgroup uncertainty assessment. "
    "Gradient boosting achieved the strongest ROC-AUC in the saved "
    "final comparison, while threshold and subgroup analyses show "
    "why a single performance metric is not sufficient."
)

add_paragraph(
    "Future work should prioritize independent external validation, "
    "documented threshold-cost assumptions, additional fairness tests "
    "and careful review of data provenance before considering any "
    "real-world use."
)

# =========================================================
# 10. REFERENCES
# =========================================================

add_heading("10. References")

add_paragraph(
    "1. Yeh, I. C., &amp; Lien, C. H. (2009). The comparisons of "
    "data mining techniques for the predictive accuracy of probability "
    "of default of credit card clients. <i>Expert Systems with "
    "Applications, 36</i>(2), 2473–2480. "
    "https://doi.org/10.1016/j.eswa.2007.12.020"
)

add_paragraph(
    "2. University of California, Irvine. Default of Credit Card "
    "Clients dataset. UCI Machine Learning Repository. "
    "https://archive.ics.uci.edu/dataset/350/default"
)

add_paragraph(
    "3. Lundberg, S. M., &amp; Lee, S. I. (2017). A Unified Approach "
    "to Interpreting Model Predictions. <i>Advances in Neural "
    "Information Processing Systems, 30.</i>"
)

add_paragraph(
    "4. Scikit-learn developers. Scikit-learn documentation: model "
    "evaluation and probability calibration. "
    "https://scikit-learn.org/stable/"
)

add_paragraph(
    "The references provide background for the dataset, model "
    "explanation method and evaluation concepts. Project metrics "
    "are taken from the locally saved experiment results.",
    "SmallNote",
)

# =========================================================
# BUILD PDF
# =========================================================

doc = SimpleDocTemplate(
    str(PDF_PATH),
    pagesize=A4,
    rightMargin=RIGHT_MARGIN,
    leftMargin=LEFT_MARGIN,
    topMargin=0.65 * inch,
    bottomMargin=0.65 * inch,
    title="Credit Risk Research Capstone",
    author="Credit Risk Research Capstone",
    subject=(
        "Predictive modelling, calibration, explainability "
        "and fairness assessment"
    ),
)

doc.build(
    story,
    onFirstPage=add_page_number,
    onLaterPages=add_page_number,
)

print()
print("=" * 60)
print("CREDIT RISK RESEARCH REPORT GENERATED")
print("=" * 60)
print(f"PDF: {PDF_PATH}")
print(f"File exists: {PDF_PATH.exists()}")

if PDF_PATH.exists():
    print(f"File size: {PDF_PATH.stat().st_size / 1024:.1f} KB")

print()
print("Please open the PDF and inspect the tables and figures.")
print("Verify that all reported results match the saved experiment files.")