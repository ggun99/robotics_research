import os
import re
import sys
import time
import glob
import subprocess

# ...existing code...
ORIG = os.path.join(os.path.dirname(__file__), "test_with_collision_fix.py")
if not os.path.exists(ORIG):
    raise FileNotFoundError(ORIG)

# 조합 설정 (원하는 값으로 수정)
obs_y_values = [0.5]
w3_values = [3.0, 5.0]

TIMEOUT = 900  # 초, 필요시 조정
PAUSE_BETWEEN = 1.0

# 찾을 블록 (시작/끝 주석을 그대로 사용)
block_re = re.compile(
    r"(#=+ 변경 변수들 =+?\n)(.*?)(\n#=+)",
    re.S
)

# 기존 OUTPUT_DIR 치환용 패턴
outdir_re = re.compile(r'OUTPUT_DIR\s*=\s*os\.path\.join\([^\n]+\)\s*')

def make_temp_text(orig_text, obs_y, w3, run_tag):
    # 1) replace the variable block
    new_block = (
        "#================================================== 변경 변수들 ==============================\n"
        f"obstacles_positions = np.array([[1.25, {float(obs_y):.3f}, 1.]])\n"
        f"w1 = 1.0 \n"
        f"w2 = 1.0\n"
        f"w3 = {float(w3):.3f}\n"
        f"w4 = 0.2\n"
        f"desired_position = [3.5,1.0,1]\n"
        "#===========================================================================================\n"
    )
    if block_re.search(orig_text):
        txt = block_re.sub(lambda m: m.group(1) + new_block + m.group(3), orig_text, count=1)
    else:
        # fallback: append block near top
        txt = new_block + "\n" + orig_text

    # 2) change OUTPUT_DIR to unique folder per run
    out_dir_name = f"collision_weight_results_w3_{str(w3).replace('.', '_')}_obsY_{str(obs_y).replace('.', '_')}_{run_tag}"
    new_out_line = f"OUTPUT_DIR = os.path.join(os.path.dirname(__file__), r\"{out_dir_name}\")\nos.makedirs(OUTPUT_DIR, exist_ok=True)\n"
    if outdir_re.search(txt):
        txt = outdir_re.sub(new_out_line, txt, count=1)
    else:
        # insert after imports block (approx)
        lines = txt.splitlines()
        insert_at = 0
        for i, L in enumerate(lines[:60]):
            if L.strip().startswith("from") or L.strip().startswith("import"):
                insert_at = i + 1
        lines.insert(insert_at, new_out_line)
        txt = "\n".join(lines)

    return txt

def run_all():
    with open(ORIG, 'r') as f:
        orig_text = f.read()

    combos = [(oy, w3) for oy in obs_y_values for w3 in w3_values]
    total = len(combos)
    for idx, (oy, w3) in enumerate(combos, start=1):
        tag = f"run{idx:03d}"
        print(f"[{idx}/{total}] obs_y={oy}, w3={w3} -> starting")
        temp_path = os.path.join(os.path.dirname(__file__), f"temp_test_{tag}.py")
        temp_text = make_temp_text(orig_text, oy, w3, tag)
        with open(temp_path, 'w') as tf:
            tf.write(temp_text)
        try:
            p = subprocess.run([sys.executable, temp_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=TIMEOUT)
            print(f"[{idx}] returncode={p.returncode}")
            if p.stdout:
                print("...last output lines:")
                for L in p.stdout.splitlines()[-20:]:
                    print(L)
        except subprocess.TimeoutExpired:
            print(f"[{idx}] TIMEOUT after {TIMEOUT}s")
        except Exception as e:
            print(f"[{idx}] ERROR: {e}")
        finally:
            try:
                os.remove(temp_path)
            except Exception:
                pass
        time.sleep(PAUSE_BETWEEN)
    print("All runs done.")

if __name__ == "__main__":
    run_all()
# ...existing code...