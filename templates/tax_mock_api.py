from flask import Flask, request, Response
import json
from tax_mock_generator import generate_mock_data
from urllib.parse import urlparse
import os
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from werkzeug.utils import secure_filename
import zipfile

app = Flask(__name__)
# Ensure Flask does not escape non-ASCII (Chinese) to \uXXXX
app.config['JSON_AS_ASCII'] = False

# set up a rotating file handler for structured logging
log_dir = Path(__file__).resolve().parent / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)
log_path = log_dir / 'tax_mock_api.log'
handler = RotatingFileHandler(str(log_path), maxBytes=5 * 1024 * 1024, backupCount=3, encoding='utf-8')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')
handler.setFormatter(formatter)
if not app.logger.handlers:
    app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)


@app.route('/infoextract', methods=['GET', 'POST'])
def generate():
    """Generate mock data. GET: use query param n_files. POST: use JSON body {"n_files":int}.
    Returns application/json; charset=utf-8 with unescaped Chinese.
    """
    n_files = 2
    if request.method == 'GET':
        try:
            n_files = int(request.args.get('n_files', 2))
        except Exception:
            n_files = 2
    else:
        body = request.get_json(silent=True) or {}
        try:
            n_files = int(body.get('n_files', 2))
        except Exception:
            n_files = 2

    # Optional remote_url handling: accept but do not fetch (mock-only behavior)
    remote_url = None
    if request.method == 'GET':
        remote_url = request.args.get('remote_url') or None
    else:
        remote_url = body.get('remote_url') if isinstance(body, dict) else None
    app.logger.info('remote_url %s', remote_url)
    # validate simple scheme if provided (only http/https allowed for now)
    def _valid_remote(u):
        if not u:
            return False
        try:
            p = urlparse(u)
            return p.scheme in ('http', 'https') and bool(p.netloc)
        except Exception:
            return False

    valid_remote = _valid_remote(remote_url)

    # If a valid remote_url is provided, determine whether it's a single file or a directory-like path.
    # Directory-like: path ends with '/' or the basename has no extension. File-like: basename contains a dot.
    remote_is_dir = False
    remote_basename = None
    if valid_remote:
        try:
            parsed_tmp = urlparse(remote_url)
            ppath = parsed_tmp.path or ''
            bname = os.path.basename(ppath)
            # consider as directory if path ends with '/' or basename has no dot
            if ppath.endswith('/') or (bname and '.' not in bname):
                remote_is_dir = True
                # folder base fallback to netloc if basename empty
                remote_basename = bname or parsed_tmp.netloc
            else:
                # single file
                remote_is_dir = False
                remote_basename = bname or parsed_tmp.netloc
                # if it's a file, override to return a single item
                n_files = 1
        except Exception:
            remote_is_dir = False
            remote_basename = None

    mock = generate_mock_data(n_files)

    # --- handle uploaded files (multipart/form-data) if present ---
    uploaded_names = []
    try:
        if request.files:
            files = request.files.getlist('files')
            if files:
                upload_dir = Path(__file__).resolve().parent / 'uploads'
                upload_dir.mkdir(parents=True, exist_ok=True)
                for f in files:
                    # secure the filename and save
                    fname = secure_filename(f.filename)
                    if not fname:
                        continue
                    dest = upload_dir / fname
                    f.save(str(dest))
                    # if zip, extract
                    if fname.lower().endswith('.zip'):
                        try:
                            with zipfile.ZipFile(dest, 'r') as zf:
                                zf.extractall(path=str(upload_dir))
                            # add extracted file names
                            for zi in zf.namelist():
                                if zi.endswith('/'):
                                    continue
                                uploaded_names.append(os.path.basename(zi))
                        except Exception:
                            app.logger.exception('zip extract failed for %s', dest)
                    else:
                        uploaded_names.append(fname)
    except Exception:
        app.logger.exception('error handling uploaded files')

    # if files were uploaded, override mock filenames and n_files
    if uploaded_names:
        # ensure mock has enough entries; regenerate if necessary
        if not mock.get('data') or len(mock.get('data', [])) < len(uploaded_names):
            mock = generate_mock_data(len(uploaded_names))
        # map uploaded names to mock entries
        for i, name in enumerate(uploaded_names):
            if i < len(mock.get('data', [])):
                mock['data'][i]['filename'] = name
                mock['data'][i].setdefault('meta', {})['uploaded_local'] = str((Path(__file__).resolve().parent / 'uploads' / name))
                mock['data'][i]['remote_url'] = None

    # attach provided remote_url info to the metadata so downstreams can see it was received
    if remote_url:
        meta = mock.setdefault('meta', {})
        meta['remote_url_provided'] = remote_url
        meta['remote_url_valid'] = bool(valid_remote)
        meta['remote_url_type'] = 'directory' if (valid_remote and remote_is_dir) else 'file'
        if 'data' in mock and isinstance(mock['data'], list):
            for entry in mock['data']:
                # default attach (None if invalid). For directories we'll set a per-entry remote_url later.
                entry['remote_url'] = remote_url if valid_remote and not remote_is_dir else None

    # If remote_url is valid, replace generated filenames appropriately
    if valid_remote and 'data' in mock and isinstance(mock['data'], list) and len(mock['data']) > 0:
        try:
            parsed = urlparse(remote_url)
            if not remote_is_dir:
                # single file: use basename (ensure extension)
                base = os.path.basename(parsed.path) or parsed.netloc
                if '.' not in base:
                    base = base + '.pdf'
                mock['data'][0]['filename'] = base
                mock['data'][0].setdefault('meta', {})['original_remote_url'] = remote_url
                # ensure entry.remote_url is the exact file URL
                mock['data'][0]['remote_url'] = remote_url
            else:
                # directory: generate filenames based on folder base and index
                folder = remote_basename or parsed.netloc
                folder = folder.replace('/', '_')
                base_url = remote_url.rstrip('/')
                for idx, entry in enumerate(mock['data'], start=1):
                    fname = f"{folder}_{idx}.pdf"
                    entry['filename'] = fname
                    entry.setdefault('meta', {})['original_remote_url'] = base_url + '/' + fname
                    # set entry.remote_url to point to the file under the provided folder
                    entry['remote_url'] = base_url + '/' + fname
        except Exception:
            # if any error, leave generated filenames as-is
            pass

    # Explicitly dump with ensure_ascii=False and include charset in Content-Type
    body = json.dumps(mock, ensure_ascii=False)
    return Response(body, mimetype='application/json; charset=utf-8')

# @app.route('/infoextract', methods=['GET', 'POST'])
# def generate():
#     """Generate mock data. GET: use query param n_files. POST: use JSON body {"n_files":int}.
#     Returns application/json; charset=utf-8 with unescaped Chinese.
#     """
#     n_files = 2
#     if request.method == 'GET':
#         try:
#             n_files = int(request.args.get('n_files', 2))
#         except Exception:
#             n_files = 2
#     else:
#         body = request.get_json(silent=True) or {}
#         try:
#             n_files = int(body.get('n_files', 2))
#         except Exception:
#             n_files = 2

#     mock = generate_mock_data(n_files)

#     # Explicitly dump with ensure_ascii=False and include charset in Content-Type
#     body = json.dumps(mock, ensure_ascii=False)
#     return Response(body, mimetype='application/json; charset=utf-8')
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001, debug=True)
