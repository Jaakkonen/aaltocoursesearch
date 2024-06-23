import os
import pandas as pd
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
import requests

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

def load_courses() -> pd.DataFrame:
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
