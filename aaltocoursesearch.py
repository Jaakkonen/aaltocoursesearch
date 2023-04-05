from collections.abc import Iterable
import os
import re
from typing import Literal, Optional
import pandas as pd
import json
import streamlit as st
from st_aggrid import AgGrid
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
from dataclasses import dataclass
import numpy as np
from streamlit.delta_generator import DeltaGenerator
import requests


@dataclass
class ColumnProperties:
    dtype: np.dtype
    uniq_vals: int


TYPES_READABLE = {
    "teaching-participation-lectures": "Lectures",
    "exam-exam": "Exam",
    "teaching-participation-small-group": "Small group",
    "independent-work-essay": "Essay",
    "thesis-bachelor": "Thesis",
    "teaching-participation-project": "Project",
    "teaching-participation-online": "Online",
    "independent-work-project": "Work project",
}

# Ordered dict of (old, new) column names.
ACTIVE_COL_NAMES = {
    "code": "Code",
    "name": "Name",
    "credits": "Credits",
    "startDate": "Start date",
    "type": "Format",
    "summary.teacherInCharge": "Responsible teacher",
    # TODO: Add calculated "duration" field?
    # Somewhat useful fields
    "endDate": "End date",
    "summary.assesmentMethods": "Assesment methods",
    "summary.literature": "Literature",
    "summary.prerequisites": "Prerequisites",
    "summary.learningOutcomes": "Learning outcomes",
    "summary.content": "Content"

}

# Default hidden fields
HIDDEN_COL_NAMES = {
    "teachers": "Teachers",
    "organizationName": "Department",
    "summary.teachingPeriod": "Period",
    "summary.level": "Level",
    "summary.registration": "Registration",
    "enrolmentStartDate": "Enrollment start",  # TODO: Add "Enrolment active" button & a date selector next to it if active
    "enrolmentEndDate": "Enrollment end",
    "summary.substitutes": "Substitutes",
    "summary.workload": "Workload",
    "summary.languageOfInstruction": "Language",
    "summary.additionalInformation": "Additional information",
    "summary.gradingScale": "Grading scale",
}

COL_NAMES = ACTIVE_COL_NAMES | HIDDEN_COL_NAMES
COL_NAMES_INV = {v: k for k, v in COL_NAMES.items()}

@st.cache_data
def load_data() -> pd.DataFrame:
    token = os.environ["AALTO_COURSES_API_TOKEN"]
    coursereals = requests.get(
        "https://course.api.aalto.fi/api/sisu/v1/courseunitrealisations",
        params={"USER_KEY": token}
    ).json()

    df = pd.json_normalize(coursereals)

    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    df["teachers"] = df["teachers"].apply(lambda x: ", ".join(x))
    df["summary.teacherInCharge"] = df["summary.teacherInCharge"].apply(
        lambda x: ", ".join(x)
    )

    # Remove always unnecessary fields
    ALWAYS_IGNORE_FIELDS = {
        "organizations",
        "studySubGroups",
        "courseUnitId",
        "id",
        "languageOfInstructionCodes",
        "organizationId",
        "summary.substitutes.courseUnits",
        "summary.cefrLevel",  # Relevant only for language courses
        "mincredits",  # Duplicates credits.min but as string
    }
    df.type = df['type'].apply(lambda x: TYPES_READABLE.get(x, x))

    return df[[col for col in df.columns if col not in ALWAYS_IGNORE_FIELDS]]


def preferred_lang_df(
    df: pd.DataFrame, preferred: Optional[Literal["fi", "sv", "en"]]
) -> pd.DataFrame:
    """
    Selects only columns ending with preferred language code (or english if preferred is not available)
    from the dataframe and renames those to not have the language code suffix.
    """
    if preferred is None:
        preferred = "en"

    localized_cols = [col for col in df.columns if col.endswith((".fi", ".sv", ".en"))]
    localized_fields = {col[:-3] for col in localized_cols}

    df = df.copy()
    for f in localized_fields:
        if f"{f}.{preferred}" in df.columns:
            df[f] = df[f + "." + preferred]
        else:
            # Try getting english instead
            df[f] = df[f + ".en"]

    return df[[col for col in df.columns if col not in localized_cols]]


@dataclass
class MinMaxField:
    mincol: str
    maxcol: str
    col: str


def get_minmax_fields(df: pd.DataFrame) -> tuple[pd.DataFrame, list[MinMaxField]]:
    """
    Gets columns that have both {col}.end and {col}.min fields.
    """
    a = []
    for col in df.columns:
        if not col.endswith(".min"):
            continue
        mincol = col
        col = col.removesuffix(".min")
        maxcol = col + ".max"
        if maxcol in df.columns and is_numeric_dtype(df[mincol]) and is_numeric_dtype(df[maxcol]):
            df[col] = df[mincol].astype(str) + "-" + df[maxcol].astype(str)
            df[col].update(df[df[maxcol] == df[mincol]][mincol].astype(str))
            df['__'+mincol] = df.pop(mincol)
            df['__'+maxcol] = df.pop(maxcol)
            a.append(MinMaxField('__'+mincol, '__'+maxcol, col))

    return df, a

def filter_df(
    df: pd.DataFrame, modification_container: DeltaGenerator,
    minmax_cols: dict[str, tuple[str, str]] = {},
    to_filter: Iterable[str] = (),
) -> pd.DataFrame:
    """
    Creates multiselect for each column and for selected columns creates
    substring/range/date selectors that are used to filter the dataframe.

    Range fields can be given in minmax_props and are not automatically detected.
    """
    df = df.copy()

    with modification_container:
        # TODO: Add name mapping

        for column in to_filter:
            # NOTE: Caller is responsible for type checking in this case
            left, right = st.columns((1, 20))
            left.write("↳")
            if (
                column in minmax_cols
            ):
                mincol = df[minmax_cols[column][0]]
                maxcol = df[minmax_cols[column][1]]
                _min = float(mincol.min())
                _max = float(maxcol.max())
                step = (_max - _min) / 100
                (minsearch, maxsearch) = right.slider(
                    f"Values for {column}",
                    _min,
                    _max,
                    (_min, _max),
                    step=step,
                )
                # Get items where search and possible credit ranges overlap
                df = df[(maxsearch >= mincol) & (minsearch <= maxcol)]
            # Treat columns with < 10 unique values as categorical
            elif is_categorical_dtype(df[column]) or df[column].nunique() < 40:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    _min,
                    _max,
                    (_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    # TODO: case-insensitive
                    # TODO: verify regex is supported
                    df = df[df[column].str.contains(user_text_input, case=False, regex=True)]

    return df




def create_view():
    st.set_page_config(layout="wide")

    df = load_data()

    st.title("Aalto course search")
    st.write(
        """\
  Explere the courses available in Aalto University and when they are offered.
  Start by adding filters for the department and checking courses starting in
  next 2 months to inspect courses in beginning period.
  """
    )
    col1, col2, col3 = st.columns([.15, .1, .7])
    lang = col1.selectbox("Field language", ("fi", "en", "sv"))
    enrolment_active = col2.checkbox("Enrollment active")

    df = preferred_lang_df(df, lang)

    df, minmax_fields = get_minmax_fields(df)
    # Create minmax fields

    # Rename & reorder fields
    for old, new in reversed(COL_NAMES.items()):
      df.insert(0, new, df.pop(old))


    filterable_columns = [col for col in df.columns if not col.startswith('__')]

    to_filter_columns = col3.multiselect("Filter on", filterable_columns)
    modification_container = st.container()

    df = filter_df(
      df,
      modification_container,
      minmax_cols={"Credits": ("__credits.min", "__credits.max")},
      to_filter=to_filter_columns
    )[:1000]

    if enrolment_active:
      now = pd.Timestamp.now()
      df = df[(df["Enrollment start"] <= now) & (now <= df["Enrollment end"])]

    # Convert datetimes to readable format
    for c in df.columns:
        if is_datetime64_any_dtype(df[c]):
            df[c] = df[c].dt.strftime("%Y-%m-%d")

    # Filter out helper fields
    df = df[[c for c in df.columns if not c.startswith('__')]]

    # TODO: Ability to select a row to get more info about it
    # https://github.com/streamlit/streamlit/issues/455
    # is required prior to that (now this could only be implemented using Dash...)
    st.dataframe(df, height=900)

    st.write("""**Hint**: Click the expand button next to the table while hovering over it to make it full screen.""")

    st.write("Please submit all any bug reports and/or feature requests to the [GitHub repository](https://github.com/Jaakkonen/aaltocoursesearch).")
    st.write("Made by [Jaakkonen](https://github.com/Jaakkonen) with extensive usage of guild room sofas 🛋️☕❤️.")



create_view()
