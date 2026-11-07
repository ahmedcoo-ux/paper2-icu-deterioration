# register(paper, NAME, path): adds  NAME = r"path"  to paths\<paper>.py if NAME is not already there
import os, re
ROOT = r"<PHD_ROOT>\paths"
def register(paper, name, path, comment=""):
    name = name.upper(); fn = os.path.join(ROOT, f"{paper}.py")
    text = open(fn, encoding="utf-8").read()
    if re.search(rf"^{re.escape(name)}\s*=", text, flags=re.M):
        return False
    line = f'{name} = r"{path}"' + (f"   # {comment}" if comment else "")
    with open(fn, "a", encoding="utf-8") as f:
        f.write(("" if text.endswith("\n") else "\n") + line + "\n")
    print(f"registered in {paper}.py: {line}")
    return True
