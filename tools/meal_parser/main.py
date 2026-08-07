"""
Meal Parser - Parse meal data files into artifact store

Simple utility tool that reads meal data from CSV or Parquet files and stores them 
in the artifact store, returning an artifact ID that can be used with other tools.

## When to Use
- Parsing meal data files before analysis
- Converting file paths to artifact IDs for tool chaining
- Quick data ingestion for meal analysis workflows

## When NOT to Use
- If you already have data in the artifact store
- For non-meal data (this validates meal data columns)
- For streaming/real-time data (this is for batch file parsing)

## Requirements
- Input: Path to CSV or Parquet file with meal data
- Required columns: collection_timestamp, collection_date, product_name,
  weight_g, calories_kcal, carbohydrate_g, protein_g, lipid_g
- Optional columns: sodium_mg, short_food_name, local_timestamp

## Example
    # Parse meal data and get artifact ID
    result = meal_parser(file_path="data/synthetic_001/diet.csv")
    logging.info(f"Artifact ID: {result.meal_aid}")

    # Chain with the diet-logging QC tool
    qc = qc_diet_data(meal_aid=result.meal_aid)
"""
from __future__ import annotations

import logging
from typing import Annotated
from pathlib import Path
from pydantic import BaseModel, Field
import pandas as pd

from pha.tool_decorator import tool
from pha.artifact_store import put_artifact
from pha.utils import DEFAULT_TIMEZONE, localize_timestamps


# ============================================================================
# Constants
# ============================================================================

_REQUIRED_COLUMNS = [
    'collection_timestamp', 'collection_date', 'product_name',
    'weight_g', 'calories_kcal', 'carbohydrate_g', 'protein_g', 'lipid_g'
]

# Optional columns filled in when absent. A string value names another column to
# copy from; a literal is used as the fill value.
#
# NOTE: sodium_mg fills with 0.0 rather than NaN. If your source data has no
# sodium column, downstream sodium means will be biased low, not missing. This
# is deliberate for tool-chaining convenience, but callers should be aware.
_OPTIONAL_COLUMNS = {
    'short_food_name': 'product_name',
    'local_timestamp': 'collection_timestamp',
    'sodium_mg': 0.0,
}

_SUPPORTED_CSV_EXTENSIONS = ('.csv',)
_SUPPORTED_PARQUET_EXTENSIONS = ('.parquet', '.pq')
_SUPPORTED_EXTENSIONS = _SUPPORTED_CSV_EXTENSIONS + _SUPPORTED_PARQUET_EXTENSIONS

_DATE_FORMAT = '%Y-%m-%d'
_ARTIFACT_PREFIX = 'meal'
_NO_DATES_LABEL = 'No dates'


# ============================================================================
# Metadata Model
# ============================================================================

class MealParserMetadata(BaseModel):
    """Output metadata for meal_parser tool.
    
    Convention: Fields ending in '_aid' are artifact references (file paths).
    All other fields are facts (light scalars for LLM reasoning).
    
    Attributes:
        meal_aid: Path to parsed meal data artifact
        file_path: Path to the original file that was parsed
        rows_parsed: Number of rows parsed from the file
        columns: List of columns in the parsed data
        date_range: Range of dates in the data (earliest to latest)
    """
    meal_aid: str = Field(
        description="Path to parsed meal data artifact"
    )
    file_path: str = Field(
        description="Path to the original file that was parsed"
    )
    rows_parsed: int = Field(
        description="Number of rows parsed from the file",
        ge=0
    )
    columns: list[str] = Field(
        description="List of columns in the parsed data",
        default_factory=list
    )
    date_range: str = Field(
        description="Date range in the data (e.g., '2024-01-01 to 2024-01-14')"
    )


# ============================================================================
# Main Tool: meal_parser
# ============================================================================

@tool(
    version="1.0.0",
    categories=["Nutrition", "DataProcessing"],
    description="Parse meal data from CSV or Parquet files into artifact store"
)
def meal_parser(
    file_path: Annotated[str, Field(
        description="Path to CSV or Parquet file containing meal data",
        min_length=1
    )],
    *,
    tz: Annotated[str, Field(
        description="Timezone for timestamp localization. Must match the value "
                    "passed to cgm_parser, or meals and glucose will not align."
    )] = DEFAULT_TIMEZONE,
) -> MealParserMetadata:
    """Parse meal data from a file and store it in the artifact store.
    
    This tool reads meal data from CSV or Parquet files, validates the required
    columns, and stores the data in the artifact store. The returned meal_aid
    can then be used with other meal analysis tools.
    
    Input Data Requirements:
        Required columns: collection_timestamp, collection_date, product_name,
                         weight_g, calories_kcal, carbohydrate_g, protein_g, lipid_g
        Optional columns: sodium_mg, short_food_name, local_timestamp
    
    Args:
        file_path: Path to the meal data file (CSV or Parquet)
        tz: Timezone for timestamp localization. Use the same value as
            cgm_parser so that meal times and glucose times align.
        
    Returns:
        MealParserMetadata with meal_aid (artifact path) and file information
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is unsupported or missing required columns
    """
    # Convert to Path object
    file_path_obj = Path(file_path)
    
    # Check if file exists
    if not file_path_obj.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Load file based on extension
    if file_path_obj.suffix.lower() in _SUPPORTED_CSV_EXTENSIONS:
        logging.debug(f"Loading CSV file: {file_path}")
        df = pd.read_csv(file_path_obj)
    elif file_path_obj.suffix.lower() in _SUPPORTED_PARQUET_EXTENSIONS:
        logging.debug(f"Loading Parquet file: {file_path}")
        df = pd.read_parquet(file_path_obj)
    else:
        supported_formats = ', '.join(_SUPPORTED_EXTENSIONS)
        raise ValueError(
            f"Unsupported file format: {file_path_obj.suffix}. "
            f"Supported formats: {supported_formats}"
        )
    
    # Validate required columns
    missing_columns = [col for col in _REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {', '.join(missing_columns)}. "
            f"Required columns are: {', '.join(_REQUIRED_COLUMNS)}"
        )
    
    logging.info(f"Successfully loaded {len(df)} rows from {file_path}")
    
    # Add optional columns with defaults if missing
    for col_name, default_value in _OPTIONAL_COLUMNS.items():
        if col_name not in df.columns:
            if isinstance(default_value, str):
                # String value means copy from another column
                df[col_name] = df[default_value]
                logging.debug(f"Added optional column '{col_name}' from '{default_value}'")
            else:
                # Literal value to use as default
                df[col_name] = default_value
                logging.debug(f"Added optional column '{col_name}' with default value {default_value}")
    
    # Normalize timestamps for cross-sensor alignment
    df['collection_timestamp'] = pd.to_datetime(df['collection_timestamp'], errors='coerce')
    df['collection_timestamp'] = localize_timestamps(df['collection_timestamp'], tz=tz)
    df['local_timestamp'] = pd.to_datetime(df['local_timestamp'], errors='coerce')
    df['local_timestamp'] = localize_timestamps(df['local_timestamp'], tz=tz)

    # Get date range
    if len(df) > 0 and 'collection_date' in df.columns:
        dates = pd.to_datetime(df['collection_date']).sort_values()
        min_date = dates.iloc[0].strftime(_DATE_FORMAT)
        max_date = dates.iloc[-1].strftime(_DATE_FORMAT)
        if min_date == max_date:
            date_range = min_date
        else:
            date_range = f"{min_date} to {max_date}"
        logging.info(f"Data date range: {date_range}")
    else:
        date_range = _NO_DATES_LABEL
        logging.warning("No dates found in data")
    
    # Store in artifact store
    meal_aid = put_artifact(df, prefix=_ARTIFACT_PREFIX)
    logging.info(f"Stored meal data with artifact ID: {meal_aid}")
    
    # Return metadata
    return MealParserMetadata(
        meal_aid=meal_aid,
        file_path=str(file_path),
        rows_parsed=len(df),
        columns=list(df.columns),
        date_range=date_range
    )

