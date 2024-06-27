import os
from pathlib import Path
import shutil

import pandas as pd
from dotenv import load_dotenv
from pyannote.audio import Pipeline
from tqdm import tqdm
import torch

###############################################################################

DATA_DIR = Path(__file__).parent.parent / "data"
FULL_METADATA_PATH = (DATA_DIR / "full-dataset-metadata.csv").absolute().resolve()
DIARIZED_TRANSCRIPTS_DIR = DATA_DIR / "diarized-transcripts"

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
        council_annotations = pd.read_csv(
            DATA_DIR / f"whole-period-seg-{short_name}.csv"
        )
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
    ].replace(
        {
            "good-safe-use": "good",
            "good-safe-to-use": "good",
            "okay-use-if-needed": "good",
            "bad-do-not-use": "bad",
        }
    )

    # Groupby council and session_id
    # check period_start_sentence_index or period_end_sentence_index
    # If either is not null, the session has a public comment period
    # Create a row that is council, session_id,
    # has_public_comment_period, and transcript_quality
    def process_council_session_group(group):
        return pd.Series(
            {
                "has_public_comment_period": (
                    group["period_start_sentence_index"].notnull().any()
                    or group["period_end_sentence_index"].notnull().any()
                ),
                "transcript_quality": group["transcript_quality"].iloc[0],
            }
        )

    council_session_annotations = (
        all_annotations.groupby(
            ["council", "session_id"],
        )
        .apply(process_council_session_group)
        .reset_index()
    )

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
    df[(df["year"] == 2022) & (df["session_datetime"].dt.month.isin([1, 2, 3]))].copy()

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

    # Prep out dir
    if DIARIZED_TRANSCRIPTS_DIR.exists():
        shutil.rmtree(DIARIZED_TRANSCRIPTS_DIR)

    DIARIZED_TRANSCRIPTS_DIR.mkdir(exist_ok=True)

    # Apply pretrained pipeline
    new_metadata_rows = []
    for _, session_details in tqdm(
        df_2021_sample.iterrows(), desc="Sessions", total=len(df_2021_sample)
    ):
        # Load transcript
        transcript = pd.read_csv(session_details["transcript_as_csv_path"])

        # Diarize
        diarization = pipeline(session_details["audio_path"])

        # For each speaker turn, combine sentences
        # from the transcript that are included within the turn together
        speaker_annotated_transcript_rows = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            # Get transcript portion via start_time and end_time
            sentences_within_turn = transcript.loc[
                (transcript["start_time"] >= turn.start)
                & (transcript["end_time"] < turn.end)
            ]

            # Combine the sentences
            combined_text = " ".join(sentences_within_turn["text"])

            # Append to speaker_annotated_transcript_rows
            speaker_annotated_transcript_rows.append(
                {
                    "start": sentences_within_turn["start_time"].min(),
                    "end": sentences_within_turn["end_time"].max(),
                    "speaker": speaker,
                    "text": combined_text,
                    "transition": '""',
                    "meeting-section": "Other",
                    "speaker-role": "Other",
                }
            )

        # Create a DataFrame from speaker_annotated_transcript_rows
        speaker_annotated_transcript = pd.DataFrame(speaker_annotated_transcript_rows)

        # Save to the same location as the
        # original transcript with "diarized-" prepended
        council = session_details["council"]
        session_id = session_details["session_id"]
        diarized_transcript_path = (
            DIARIZED_TRANSCRIPTS_DIR / f"diarized-{council}-session-{session_id}.csv"
        )
        speaker_annotated_transcript.to_csv(
            diarized_transcript_path,
            index=False,
        )

        # Append new metadata row
        new_metadata_rows.append(
            {
                "council": session_details["council"],
                "session_id": session_details["session_id"],
                "session_datetime": session_details["session_datetime"],
                "body_name": session_details["body_name"],
                "normalized_body_name": session_details["normalized_body_name"],
                "cdp_url": session_details["cdp_url"],
                "minutes_pdf_url": session_details["minutes_pdf_url"],
            }
        )


if __name__ == "__main__":
    load_dotenv()
    main()
