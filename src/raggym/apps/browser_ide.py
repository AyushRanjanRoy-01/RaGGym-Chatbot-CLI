"""In-browser practice IDE: Monaco editor + Pyodide (pytest runs client-side).

The learner edits the solution and clicks "Run tests"; Python executes entirely
in the browser via Pyodide/WebAssembly — no untrusted code runs on the server,
and there is no server cost. Exercises are standard-library only, which Pyodide
handles natively.
"""

from __future__ import annotations

import json

# Physical lines kept <=100 chars so ruff (E501) is happy inside the template.
_HTML_TEMPLATE = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<style>
  body { margin: 0; font-family: ui-sans-serif, system-ui, sans-serif;
         background: #0d1117; color: #e6edf3; }
  #bar { display: flex; align-items: center; gap: 10px;
         padding: 8px 10px; border-bottom: 1px solid #21262d; }
  #run { background: #238636; color: #fff; border: 0; border-radius: 6px;
         padding: 6px 14px; font-weight: 600; cursor: pointer; }
  #run:disabled { opacity: .6; cursor: default; }
  #title { font-weight: 700; }
  #status { font-size: 13px; color: #8b949e; margin-left: auto; }
  #editor { height: 340px; width: 100%; }
  #out { white-space: pre-wrap; margin: 0;
         font-family: ui-monospace, Menlo, monospace; font-size: 12.5px;
         padding: 10px; background: #010409; border-top: 1px solid #21262d;
         height: 150px; overflow: auto; }
  .ok { color: #3fb950; }
  .bad { color: #f85149; }
</style>
</head>
<body>
  <div id="bar">
    <span id="title"></span>
    <button id="run" disabled>Loading…</button>
    <span id="status">booting Python (Pyodide)…</span>
  </div>
  <div id="editor"></div>
  <pre id="out">Edit the solution, then Run tests.</pre>

  <script src="https://cdn.jsdelivr.net/npm/monaco-editor@0.52.2/min/vs/loader.js"></script>
  <script src="https://cdn.jsdelivr.net/pyodide/v0.26.2/full/pyodide.js"></script>
  <script>
    const STARTER = %%STARTER%%;
    const TESTS = %%TESTS%%;
    const TITLE = %%TITLE%%;
    document.getElementById('title').textContent = TITLE;
    let editor, py;

    require.config({ paths: { vs: 'https://cdn.jsdelivr.net/npm/monaco-editor@0.52.2/min/vs' } });
    require(['vs/editor/editor.main'], function () {
      editor = monaco.editor.create(document.getElementById('editor'), {
        value: STARTER, language: 'python', theme: 'vs-dark',
        automaticLayout: true, minimap: { enabled: false }, fontSize: 13
      });
    });

    async function boot() {
      py = await loadPyodide();
      await py.loadPackage('micropip');
      const micropip = py.pyimport('micropip');
      await micropip.install('pytest');
      const run = document.getElementById('run');
      run.disabled = false; run.textContent = 'Run tests';
      document.getElementById('status').textContent = 'ready · runs in your browser';
    }
    boot().catch(function (e) {
      document.getElementById('status').textContent = 'boot failed: ' + e;
    });

    document.getElementById('run').addEventListener('click', async function () {
      const out = document.getElementById('out');
      const run = document.getElementById('run');
      run.disabled = true; out.className = ''; out.textContent = 'Running…';
      let buf = '';
      py.setStdout({ batched: function (s) { buf += s + '\n'; } });
      py.setStderr({ batched: function (s) { buf += s + '\n'; } });
      py.FS.writeFile('/home/pyodide/solution.py', editor.getValue());
      py.FS.writeFile('/home/pyodide/test_exercise.py', TESTS);
      let rc = 1;
      try {
        rc = await py.runPythonAsync([
          "import sys, os, importlib",
          "sys.path.insert(0, '/home/pyodide')",
          "os.chdir('/home/pyodide')",
          "for _m in ('solution', 'test_exercise'): sys.modules.pop(_m, None)",
          "importlib.invalidate_caches()",
          "import pytest",
          "int(pytest.main(['-q', 'test_exercise.py']))"
        ].join('\n'));
      } catch (e) { buf += String(e); }
      const passed = (rc === 0);
      out.className = passed ? 'ok' : 'bad';
      out.textContent = (passed ? '✅ PASSED\n\n' : '❌ FAILED\n\n') + (buf || '(no output)');
      run.disabled = false;
    });
  </script>
</body>
</html>
"""


def build_ide_html(starter_code: str, test_code: str, *, title: str = "Exercise") -> str:
    """Return a self-contained HTML page embedding the editor + Pyodide runner."""
    return (
        _HTML_TEMPLATE.replace("%%STARTER%%", json.dumps(starter_code))
        .replace("%%TESTS%%", json.dumps(test_code))
        .replace("%%TITLE%%", json.dumps(title))
    )


def render_browser_ide(
    starter_code: str, test_code: str, *, title: str = "Exercise", height: int = 560
) -> None:
    """Render the browser IDE inside a Streamlit app."""
    import streamlit.components.v1 as components

    components.html(build_ide_html(starter_code, test_code, title=title), height=height)
