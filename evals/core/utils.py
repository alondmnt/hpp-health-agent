"""Shared utilities for evaluation modules.

Loads reports, extracts metrics from report text, and computes error against
ground truth.
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, Optional


def load_report(report_path: str) -> str:
    """
    Load report text from PDF or Markdown file
    
    Args:
        report_path: Path to report file
        
    Returns:
        Report text as string
    """
    path = Path(report_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Report not found: {report_path}")
    
    if path.suffix.lower() == '.pdf':
        return load_pdf(report_path)
    elif path.suffix.lower() in ['.md', '.txt']:
        return load_text(report_path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")


def load_pdf(pdf_path: str) -> str:
    """Extract text from a PDF report.

    This repository generates markdown, so PyPDF2 is an optional dependency
    imported only when a PDF is actually passed.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Extracted text
    """
    try:
        import PyPDF2
    except ImportError as e:
        raise ImportError(
            "Reading PDF reports requires PyPDF2 (pip install PyPDF2). "
            "This repository generates markdown reports, so this is only "
            "needed if you point the runner at a PDF."
        ) from e

    text = []
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text.append(page.extract_text())
    return "\n".join(text)


def load_text(text_path: str) -> str:
    """Load text from markdown or text file"""
    try:
        with open(text_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        raise RuntimeError(f"Failed to load text file: {e}")


def load_ground_truth(gt_path: str) -> Dict[str, float]:
    """
    Load ground truth metrics from JSON file
    
    Args:
        gt_path: Path to ground truth JSON
        
    Returns:
        Dictionary mapping metric names to values
    """
    try:
        with open(gt_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load ground truth: {e}")


def extract_cgm_in_range_70_180(text: str) -> Optional[float]:
    """Extract Time in Range from same-line table or prose contexts."""
    patterns = [
        r'\|\s*(?:\*\*)?(?:time\s+in\s+range|in\s+range|tir)(?:\*\*)?\s*(?:\(?70[-–]180[^|]*)?\|\s*(\d+\.?\d*)\s*%?\s*(?:\||$)',
        r'\|\s*(?:\*\*)?70[-–]180[^|]*(?:\*\*)?\s*\|\s*(\d+\.?\d*)\s*%?\s*(?:\||$)',
        r'(?:TIR|time\s+in\s+range|in\s+range)\s*(?:\(?70[-–]180[^)\n|]*\)?)?\s*(?:[:=≈~]|\bis\b)\s*(\d+\.?\d*)\s*%',
        r'(\d+\.?\d*)\s*%\s*(?:of\s+readings\s+)?(?:in\s+(?:the\s+)?(?:healthy\s+)?range|time\s+in\s+range)',
    ]

    for line in text.splitlines():
        for pattern in patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue

    return None


def extract_metric_patterns(text: str) -> Dict[str, Optional[float]]:
    """
    Extract common CGM metrics using pattern matching.

    This is a simple regex-based extractor. For better accuracy,
    use DSPy or LLM-based extraction.

    Args:
        text: Report text

    Returns:
        Dictionary of metric names (cgm_* standard) to extracted values
    """
    # Output keys use cgm_* standard names (matching ground_truth.json)
    metrics = {
        'cgm_mean': None,
        'cgm_cv': None,
        'cgm_in_range_70_180': None,
        'cgm_ea1c': None,
        'cgm_gmi': None,
        'cgm_adrr': None,
        'cgm_lbgi': None,
        'cgm_hbgi': None,
    }

    # Patterns for each metric - handles markdown tables and prose formats
    # Table format: | **Label** | value unit |
    # Prose format: **Label:** value unit
    patterns = {
        'cgm_mean': [
            # Table: | **Mean Glucose** | 93.0 mg/dL |
            r'\*\*(?:Mean|Average)\s+Glucose\*\*[^|]*\|\s*(\d+\.?\d*)\s*mg/dL',
            # Prose: **Mean Glucose:** 93.0 mg/dL
            r'\*\*(?:Mean|Average)\s+Glucose[^*]*\*\*[:\s]+(\d+\.?\d*)\s*mg/dL',
            # Plain table: | Mean glucose | 97.76 | mg/dL |
            r'\|\s*(?:Mean|Average)\s+glucose\s*\|\s*(\d+\.?\d*)\s*\|',
            # Prose: "glucose averaged 97.76 mg/dL" or "mean glucose of 97.76 mg/dL"
            r'(?:glucose\s+averaged|mean\s+glucose\s+of)\s+(\d+\.?\d*)\s*mg/dL',
            # Plain text fallback
            r'(?:mean|average)\s+glucose\s*(?:\([^)]*\))?\s*[:≈~\s]+(\d+\.?\d*)\s*mg/dL',
        ],
        'cgm_cv': [
            # Table: | **Coefficient of Variation (CV)** | 16.6% |
            r'\*\*(?:Coefficient\s+of\s+Variation|CV|Glucose\s+Variability)[^|]*\|\s*(\d+\.?\d*)\s*%',
            # Prose: **Glucose Variability (CV):** 12.4%
            r'\*\*(?:Coefficient\s+of\s+Variation|CV|Glucose\s+Variability)[^*]*\*\*[:\s]+(\d+\.?\d*)\s*%',
            # Plain table: | Coefficient of variation (CV) | 10.85 | % |
            r'\|\s*Coefficient\s+of\s+variation\s*\(CV\)\s*\|\s*(\d+\.?\d*)\s*\|',
            # Plain text: Coefficient of Variation (CV): 16.6%
            r'Coefficient\s+of\s+Variation\s*\(CV\)\s*:\s*(\d+\.?\d*)\s*%',
            # Prose: "CV of 10.85%" or "coefficient of variation of 10.85%"
            r'(?:CV|coefficient\s+of\s+variation)\s+of\s+(\d+\.?\d*)\s*%',
            # Plain text fallback
            r'CV\s*(?:\([^)]*\))?\s*[:≈~\s]+(\d+\.?\d*)\s*%',
        ],
        'cgm_in_range_70_180': [
            # Table: | **In Range (70-180 mg/dL)** | 97.3% |
            r'\*\*(?:In\s+Range|Time\s+in\s+Range)[^|]*70[-–]180[^|]*\|\s*(\d+\.?\d*)\s*%',
            # Table variant: | **70-180 mg/dL (Target)** | 99.84% |
            r'\*\*70[-–]180[^|]*\|\s*(\d+\.?\d*)\s*%',
            # Prose: **Time in Range (70-180 mg/dL):** 97.35%
            r'\*\*(?:In\s+Range|Time\s+in\s+Range)[^*]*70[-–]180[^*]*\*\*[:\s]+(\d+\.?\d*)\s*%',
            # Plain table: | Time in range (70-180 mg/dL) | 99.84 | % |
            r'\|\s*Time\s+in\s+range\s*\(70[-–]180[^)]*\)\s*\|\s*(\d+\.?\d*)\s*%?\s*\|',
            # Plain text fallback
            r'(?:TIR|time\s+in\s+range)\s*\(?70[-–]180[^:\n|]*[:≈~\s]+(\d+\.?\d*)\s*%',
        ],
        'cgm_ea1c': [
            # Table: | **eA1c** | 4.9% |
            r'\*\*(?:eA1c|estimated\s+A1c)[^|]*\|\s*(\d+\.?\d*)\s*%',
            # Prose: **eA1c:** 4.9%
            r'\*\*(?:eA1c|estimated\s+A1c)[^*]*\*\*[:\s]+(\d+\.?\d*)\s*%',
            # Plain text fallback
            r'eA1c\s*[:≈~\s]+(\d+\.?\d*)\s*%',
        ],
        'cgm_gmi': [
            # Table: | **Glucose Management Indicator (GMI)** | 5.5% |
            r'\*\*(?:Glucose\s+Management\s+Indicator|GMI)[^|]*\|\s*(\d+\.?\d*)\s*%',
            # Prose: **GMI (est. A1c):** 5.65%
            r'\*\*(?:GMI|Glucose\s+Management\s+Indicator)[^*]*\*\*[:\s]+(\d+\.?\d*)\s*%',
            # Plain text: Glucose Management Indicator (GMI): 5.5%
            r'Glucose\s+Management\s+Indicator\s*\(GMI\)\s*:\s*(\d+\.?\d*)\s*%',
            # Plain text fallback
            r'GMI\s*[:≈~\s]+(\d+\.?\d*)\s*%',
        ],
        'cgm_adrr': [
            r'\*\*ADRR\*\*[^|]*\|\s*(\d+\.?\d*)',
            r'ADRR\s*[:≈~\s]+(\d+\.?\d*)',
        ],
        'cgm_lbgi': [
            r'\*\*LBGI\*\*[^|]*\|\s*(\d+\.?\d*)',
            r'LBGI\s*[:≈~\s]+(\d+\.?\d*)',
        ],
        'cgm_hbgi': [
            r'\*\*HBGI\*\*[^|]*\|\s*(\d+\.?\d*)',
            r'HBGI\s*[:≈~\s]+(\d+\.?\d*)',
        ],
    }
    
    # Generic markdown-table fallbacks.
    #
    # The specific patterns above assume a two-column table holding a bare
    # number. Real reports routinely use a three-column table with the unit in
    # the value cell, e.g.
    #     | Mean glucose | 116.49 mg/dL | Fasting: 70-100 mg/dL |
    # which none of the patterns above match. Missing that is a false negative
    # in the eval, not a fault in the report, so these fallbacks take the first
    # number out of the value cell and tolerate bold markers, trailing units,
    # and any number of extra columns.
    table_labels = {
        'cgm_mean': r'(?:Mean|Average)\s+Glucose',
        'cgm_cv': r'(?:Coefficient\s+of\s+[Vv]ariation(?:\s*\(CV\))?|CV(?:\s*\([^)]*\))?)',
        'cgm_in_range_70_180': r'(?:Time\s+in\s+[Rr]ange|In\s+[Rr]ange|TIR)',
        'cgm_ea1c': r'(?:eA1c|estimated\s+A1c)',
        'cgm_gmi': r'(?:Glucose\s+Management\s+Indicator(?:\s*\(GMI\))?|GMI(?:\s*\([^)]*\))?)',
        'cgm_adrr': r'ADRR',
        'cgm_lbgi': r'LBGI',
        'cgm_hbgi': r'HBGI',
    }
    for metric, label in table_labels.items():
        patterns[metric].append(
            r'\|\s*(?:\*\*)?\s*' + label + r'\s*(?:\*\*)?[^|]*\|\s*(?:\*\*)?\s*(\d+\.?\d*)'
        )

    # Compressed scorecard fallbacks.
    #
    # Short "scorecard" style reports pack several metrics into one table cell
    # with no separator, e.g.
    #     | Glucose control | **F** | Mean 157 mg/dL - GMI 7.1% - TIR 76% | ...
    # Every pattern above expects a separator (":", "=", "is") or a cell of its
    # own, so all three values are missed and the eval reports metrics as
    # absent when the report does state them. These fallbacks accept a bare
    # label followed by the number.
    compressed_labels = {
        'cgm_mean': r'(?:Mean|Average)(?:\s+glucose)?\s+(\d+\.?\d*)\s*mg/dL',
        'cgm_in_range_70_180': r'TIR\s+(\d+\.?\d*)\s*%',
        'cgm_cv': r'CV\s+(\d+\.?\d*)\s*%',
        'cgm_gmi': r'GMI\s+(\d+\.?\d*)\s*%',
    }
    for metric, pattern in compressed_labels.items():
        patterns[metric].append(pattern)

    # The canonical anchor table wins over anything earlier in the document.
    #
    # Reports legitimately round in prose ("93% of readings") while the anchor
    # table carries full precision ("93.45%"). A top-down scan finds the rounded
    # prose value first and reports a transcription error against a correct
    # report, so the anchor row is tried before anything else.
    for metric, label in table_labels.items():
        # The bold label may carry a parenthetical, e.g.
        #   | **Time in Range (70-180 mg/dL)** | 93.45% |
        # so allow any non-pipe text inside the bold markers before closing.
        anchor = re.search(
            r'\|\s*\*\*\s*' + label + r'[^*|]*\*\*\s*\|\s*(\d+\.?\d*)', text,
            re.IGNORECASE,
        )
        if anchor:
            try:
                metrics[metric] = float(anchor.group(1))
            except ValueError:
                pass

    # Line-scoped time-in-range scan, if the anchor table did not supply it
    if metrics['cgm_in_range_70_180'] is None:
        metrics['cgm_in_range_70_180'] = extract_cgm_in_range_70_180(text)

    # Try each pattern for each metric
    for metric, pattern_list in patterns.items():
        if metrics[metric] is not None:
            continue
        for pattern in pattern_list:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    metrics[metric] = float(match.group(1))
                    break  # Found a match, move to next metric
                except ValueError:
                    continue
    
    return metrics


def calculate_mae(extracted: Dict[str, float], ground_truth: Dict[str, float]) -> Dict[str, Any]:
    """
    Calculate Mean Absolute Error between extracted and ground truth metrics
    
    Args:
        extracted: Extracted metrics from report
        ground_truth: Ground truth metrics
        
    Returns:
        Dictionary with MAE statistics and per-metric errors
    """
    errors = {}
    
    for metric, gt_value in ground_truth.items():
        if metric in extracted and extracted[metric] is not None:
            error = abs(extracted[metric] - gt_value)
            errors[metric] = {
                'extracted': extracted[metric],
                'ground_truth': gt_value,
                'error': error,
                'percent_error': (error / gt_value * 100) if gt_value != 0 else 0
            }
        else:
            errors[metric] = {
                'extracted': None,
                'ground_truth': gt_value,
                'error': None,
                'percent_error': None,
                'missing': True
            }
    
    # Calculate overall MAE (only for metrics that were extracted)
    valid_errors = [e['error'] for e in errors.values() if e.get('error') is not None]
    overall_mae = sum(valid_errors) / len(valid_errors) if valid_errors else None
    
    return {
        'overall_mae': overall_mae,
        'per_metric': errors,
        'metrics_extracted': len(valid_errors),
        'metrics_total': len(ground_truth)
    }
