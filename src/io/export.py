import io
import json


def export_csv(df):
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def export_json(df):
    records = df.to_dict(orient="records")
    return json.dumps(records, indent=2).encode("utf-8")
