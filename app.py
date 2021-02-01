import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import sample
import cv2
from shutil import copyfile

UPLOAD_FOLDER = "./test_dataset/"
ALLOWED_EXTENSIONS = set(["png", "jpg", "jpeg"])

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


class TestObject:
    def __init__(self):
        self.scale = 2
        self.ckpt_path = "./checkpoint/BIG_MODEL_487000.pth"
        self.test_data_dir = "./test_dataset"
        self.group = 1
        self.model = "carn"
        self.shave = 20
        self.sample_dir = "./result"
        self.cuda = False


def clear_folders():

    folders = ["result", "test_dataset", "static"]

    for item in folders:
        path = os.path.join(item)
        files = os.listdir(path)
        for f in files:
            os.remove(os.path.join(path, f))


def get_bicubic(img):
    fn = img.split("/")[-1].replace("LR", "LR_bicubic")
    img = cv2.imread(img)
    resized = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join("./static", fn), resized)
    return fn


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def upload_file():
    clear_folders()
    if request.method == "POST":
        # check if the post request has the file part
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files["file"]
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # If "LR" not in name we add it for convenience
            filename_without_ext, extension = (
                os.path.splitext(filename)[0],
                os.path.splitext(filename)[1],
            )
            filename = (
                filename_without_ext + "_LR" + extension
                if "LR" not in filename
                else filename
            )
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            return redirect(url_for("show_result", fn=filename))

    return """
    <!doctype html>
    <title>SR Prototype</title>
    <h1>Upload your LR text image</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    """


@app.route("/get_result/<fn>")
def show_result(fn):
    cfg = TestObject()
    sample.main(cfg)
    LR_path = os.path.join(UPLOAD_FOLDER, fn)
    BICUBIC = get_bicubic(LR_path)
    copyfile(LR_path, os.path.join("./static", fn))
    SR = os.path.join(fn.replace("LR", "SR"))
    LR = os.path.join(fn)
    print(f"SR_PATH: {SR}")
    print(f"LR_PATH: {LR}")
    print(f"BICUBIC_PATH: {BICUBIC}")
    return render_template("result.html", SR_name=SR, LR_name=LR, BICUBIC_name=BICUBIC)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
