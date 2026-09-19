import os
import shutil
import tempfile
import uuid
from pathlib import Path

from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename

from optimizer import REQUIRED_FILES, run_optimizer

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024


@app.get("/")
def home():
    return render_template("index.html", required_files=REQUIRED_FILES)


@app.post("/optimize")
def optimize():
    scenario = request.form.get("scenario", "").upper().strip()
    uploads = request.files.getlist("files")

    if scenario not in {"A", "B", "C"}:
        return render_template("index.html", required_files=REQUIRED_FILES,
                               error="Choose Scenario A, B, or C."), 400

    if len(uploads) != 8:
        return render_template("index.html", required_files=REQUIRED_FILES,
                               error="Upload exactly 8 CSV files."), 400

    work_dir = Path(tempfile.mkdtemp(prefix="trackopt_"))
    try:
        uploaded_names = set()
        for f in uploads:
            name = secure_filename(f.filename)
            if not name:
                continue
            uploaded_names.add(name)
            f.save(work_dir / name)

        missing = sorted(set(REQUIRED_FILES) - uploaded_names)
        unexpected = sorted(uploaded_names - set(REQUIRED_FILES))
        if missing or unexpected:
            parts = []
            if missing:
                parts.append("Missing: " + ", ".join(missing))
            if unexpected:
                parts.append("Unexpected: " + ", ".join(unexpected))
            return render_template("index.html", required_files=REQUIRED_FILES,
                                   error=". ".join(parts)), 400

        output_dir, log = run_optimizer(work_dir, scenario)

        zip_base = work_dir / f"track_schedule_{scenario}_{uuid.uuid4().hex[:8]}"
        zip_path = Path(shutil.make_archive(str(zip_base), "zip", output_dir))

        response = send_file(
            zip_path,
            mimetype="application/zip",
            as_attachment=True,
            download_name=f"scenario_{scenario}_outputs.zip",
        )
        response.call_on_close(lambda: shutil.rmtree(work_dir, ignore_errors=True))
        return response

    except Exception as exc:
        shutil.rmtree(work_dir, ignore_errors=True)
        return render_template(
            "index.html",
            required_files=REQUIRED_FILES,
            error=str(exc),
        ), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
