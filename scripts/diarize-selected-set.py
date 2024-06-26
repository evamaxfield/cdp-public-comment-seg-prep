import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pyannote.audio import Pipeline
from tqdm import tqdm
import torch

###############################################################################

DATA_DIR = Path(__file__).parent.parent / "data"
FULL_METADATA_PATH = (DATA_DIR / "full-dataset-metadata.csv").absolute().resolve()

###############################################################################

def _prep_dataset(
    df: pd.DataFrame,
) -> pd.DataFrame:
    # Read in the annotation files
    annotation_details = []
    for short_name in [
        "seattle",
        "oakland",
        "richmond",
    ]:
        council_annotations = pd.read_csv(DATA_DIR / f"whole-period-seg-{short_name}.csv")
        council_annotations["council"] = short_name

        # Handle seattle not having a transcript quality column
        if short_name == "seattle":
            council_annotations["transcript_quality"] = "good"

        annotation_details.append(council_annotations)

    # Combine the annotations
    all_annotations = pd.concat(annotation_details)

    # Replace transcript quality values
    all_annotations["transcript_quality"] = all_annotations[
        "transcript_quality"
        ].replace({
            "good-safe-use": "good",
            "good-safe-to-use": "good",
            "okay-use-if-needed": "good",
            "bad-do-not-use": "bad",
        })

    # Groupby council and session_id
    # check period_start_sentence_index or period_end_sentence_index
    # If either is not null, the session has a public comment period
    # Create a row that is council, session_id,
    # has_public_comment_period, and transcript_quality
    def process_council_session_group(group):
        return pd.Series({
            "has_public_comment_period": (
                group["period_start_sentence_index"].notnull().any()
                or group["period_end_sentence_index"].notnull().any()
            ),
            "transcript_quality": group["transcript_quality"].iloc[0],
        })

    council_session_annotations = all_annotations.groupby(
        ["council", "session_id"],
    ).apply(process_council_session_group).reset_index()

    # Merge the metadata with the annotations
    df = df.merge(council_session_annotations, on=["council", "session_id"])

    # Drop anything with bad transcript quality
    df = df[(df["transcript_quality"] == "good") & (df["has_public_comment_period"])]

    # Convert session_datetime to a datetime object
    df["session_datetime"] = pd.to_datetime(df["session_datetime"])
    df["year"] = df["session_datetime"].dt.year

    return df


def main() -> None:
    # Read in the metadata file
    df = pd.read_csv(FULL_METADATA_PATH)

    # Prep the dataset
    df = _prep_dataset(df)

    # Get 2021 data
    df_2021 = df[df["year"] == 2021].copy()

    # Get 2022 Jan, Feb, March data
    df_2022 = df[
        (df["year"] == 2022)
        & (df["session_datetime"].dt.month.isin([1, 2, 3]))
    ].copy()

    # Take a sample of 2021 for testing
    df_2021_sample = df_2021.sample(5)

    # Init pipeline
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=os.getenv("HF_AUTH_TOKEN"),
    )

    # Try loading pipelines to devices
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device: {device}")
    pipeline.to(torch.device(device))

    # Apply pretrained pipeline
    for _, session_details in tqdm(
        df_2021_sample.iterrows(),
        desc="Sessions",
        total=len(df_2021_sample)
    ):
        # Print results
        diarization = pipeline(session_details["audio_path"])

        # Log results
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")

        print("\n\n")
        print("-" * 80)
        print("\n\n")

    
if __name__ == "__main__":
    load_dotenv()
    main()