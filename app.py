from pathlib import Path

from flask import Flask, render_template, request, send_file, flash, redirect, url_for
from werkzeug.utils import secure_filename

from videorun import run_detection

app = Flask(__name__)
app.secret_key = "dev"
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024

BASE = Path(__file__).resolve().parent
UPLOAD_DIR = BASE / "uploads"
OUTPUT_DIR = BASE / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

ALLOWED = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    file = request.files.get("video")
    if not file or not file.filename:
        flash("No video selected")
        return redirect(url_for("index"))

    name = secure_filename(file.filename)
    if Path(name).suffix.lower() not in ALLOWED:
        flash("Use mp4, avi, mov, mkv, or webm")
        return redirect(url_for("index"))

    in_path = UPLOAD_DIR / name
    out_path = OUTPUT_DIR / f"detected_{Path(name).stem}.mp4"
    file.save(in_path)

    run_detection(str(in_path), str(out_path))

    return send_file(out_path, as_attachment=True, download_name=out_path.name)



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5900)
