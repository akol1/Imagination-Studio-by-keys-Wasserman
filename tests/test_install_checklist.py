"""The checklist at the start of the one-click install (scripts/install-checklist.sh): python3 -m pytest
tests/test_install_checklist.py. It is driven through STUDIO_CHECKLIST_INPUT and a pseudo-terminal; nothing installs."""
import os
import pty
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
LIB = ROOT / "scripts" / "install-checklist.sh"
BASH = "/bin/bash" if Path("/bin/bash").exists() else "bash"  # macOS's is bash 3.2: the list must run on it
SPARK_OPTIONAL = "--motion --voice --music --sound --bolt --train --ref2va --h3-base --image-edit --flux3 --moss --yue2 --swap"


def env(**extra):
    out = {k: v for k, v in os.environ.items() if k not in ("CI", "STUDIO_NO_CHECKLIST", "STUDIO_CHECKLIST_INPUT")}
    out.update(extra)
    return out


def run_list(answers, rows="$(checklist_rows_cuda aarch64)", free=800):
    """The flags the list prints for these answers (one per line), and what it drew."""
    script = f'. "{LIB}"; checklist_run "{rows}" {free} "Test machine" "bash oneshot-setup-spark.sh"'
    done = subprocess.run([BASH, "-c", script], capture_output=True, text=True, timeout=20,
                          env=env(STUDIO_CHECKLIST_INPUT="\\n".join(answers) + "\\n"))
    return done.returncode, done.stdout.strip(), done.stderr


def wanted(*args, **extra):
    script = f'. "{LIB}"; checklist_wanted "$@"'
    return subprocess.run([BASH, "-c", script, "x", *args], capture_output=True, timeout=20, env=env(**extra)).returncode == 0


def test_enter_installs_everything_ticked():
    code, flags, drawn = run_list([""])
    assert code == 0 and flags == SPARK_OPTIONAL
    assert "Included" in drawn and "Optional, larger" in drawn and "Free on this disk: 800 GB" in drawn
    assert "To install the same way without the list: bash oneshot-setup-spark.sh " + SPARK_OPTIONAL in drawn
    assert re.search(r"\b10 +\[x\] Training", drawn)


def test_unticking_two_rows_and_a_range():
    assert run_list(["7 8", ""])[1] == SPARK_OPTIONAL.replace("--music --sound ", "")
    assert run_list(["13-15", ""])[1] == "--motion --voice --music --sound --bolt --train --ref2va --h3-base --yue2 --swap"
    assert run_list(["2 3", ""])[1] == "--no-graphics --no-finish " + SPARK_OPTIONAL


def test_n_leaves_the_plain_install_and_a_and_d_tick_everything():
    code, flags, drawn = run_list(["n", ""])
    assert code == 0 and flags == ""
    assert "To install the same way without the list: bash oneshot-setup-spark.sh --yes" in drawn
    assert run_list(["n", "7", ""])[1] == "--music"
    assert run_list(["n", "4", "a", ""])[1] == SPARK_OPTIONAL
    assert run_list(["n", "2", "d", ""])[1] == SPARK_OPTIONAL


def test_a_bad_answer_says_why_and_changes_nothing():
    code, flags, drawn = run_list(["99", "7 x", "9-5", "a 3", ""])
    assert code == 0 and flags == SPARK_OPTIONAL
    assert "there is no row 99" in drawn and '"x" is not a number' in drawn
    assert "range low to high" in drawn and "on their own" in drawn


def test_the_core_stays_and_a_needed_row_follows():
    code, flags, drawn = run_list(["1", ""])
    assert flags == SPARK_OPTIONAL and "is always installed" in drawn
    assert run_list(["12", ""])[1] == "--motion --voice --music --sound --bolt --ref2va --image-edit --flux3 --moss --yue2 --swap"  # training goes with H3 base
    assert run_list(["n", "10", ""])[1] == "--train --h3-base"


def test_no_answer_installs_nothing():
    script = f'. "{LIB}"; checklist_run "$(checklist_rows_cuda aarch64)" 800 T C'
    done = subprocess.run([BASH, "-c", script], capture_output=True, text=True, timeout=20,
                          env=env(STUDIO_CHECKLIST_INPUT="7"))  # the input ends before an Enter
    assert done.returncode != 0 and done.stdout == "" and "nothing was installed" in done.stderr


def test_too_little_disk_is_said():
    assert "more than the free space" in run_list([""], free=100)[2]


def test_each_platform_lists_only_its_parts():
    def rows(call):
        out = subprocess.run([BASH, "-c", f'. "{LIB}"; {call}'], capture_output=True, text=True, timeout=20).stdout
        return {line.split("|")[0]: line.split("|") for line in out.splitlines()}
    spark, rtx = rows("checklist_rows_cuda aarch64"), rows("checklist_rows_cuda x86_64")
    small, big = rows('checklist_rows_mac 64 "Apple M5 Max"'), rows('checklist_rows_mac 256 "Apple M5 Ultra"')
    assert "train" in spark and "train" not in big and "turbo" not in spark
    assert "Tesseract runs on your Mac" in spark["finish"][-1] and "Tesseract runs" not in rtx["finish"][-1]
    assert not {"turbo", "ref2va", "upscale", "remove"} & set(small) and {"turbo", "ref2va", "upscale", "remove"} <= set(big)
    assert big["upscale"][7] == "ref2va"
    assert rows('checklist_rows_mac 64 "Apple M4 Max"')["core"][4] == "212000"  # 8-bit FastH3 weights too
    # every flag a row gives is one its installer reads
    for table, script in ((spark, "oneshot-setup-cuda.sh"), (big, "oneshot-setup.sh")):
        text = (ROOT / script).read_text()
        for row in table.values():
            for flag in filter(None, row[5:7]):
                assert re.search(rf"(^|[\s|]){re.escape(flag)}(\)|\|)", text, re.M), (script, flag)
            assert row[1] in ("in", "opt") and row[2] in ("0", "1") and row[4].isdigit()


def test_flags_yes_verify_upgrade_ci_and_no_terminal_leave_the_list_out():
    hook = {"STUDIO_CHECKLIST_INPUT": "\\n"}
    assert wanted(**hook) and wanted("--app", "--no-render", **hook)
    for args in (["--yes"], ["-y"], ["--music"], ["--verify"], ["--upgrade"], ["--no-finish"], ["--all"], ["--app", "--ref2va"]):
        assert not wanted(*args, **hook), args
    assert not wanted(CI="true", **hook) and not wanted(STUDIO_NO_CHECKLIST="1", **hook)
    assert not wanted()  # pipes here: no terminal


def in_terminal(script, extra=None):
    """Run a bash line with stdin, stdout and stderr on a pseudo-terminal; what it wrote there."""
    leader, follower = pty.openpty()
    proc = subprocess.Popen([BASH, "-c", script], stdin=follower, stdout=follower, stderr=follower,
                            env=env(**(extra or {})), start_new_session=True)
    os.close(follower)
    seen = b""
    sent = False
    while True:
        try:
            chunk = os.read(leader, 4096)
        except OSError:
            break
        if not chunk:
            break
        seen += chunk
        if not sent and b"> " in seen:
            os.write(leader, b"7\r")
            os.write(leader, b"\r")
            sent = True
    proc.wait(timeout=20)
    os.close(leader)
    return proc.returncode, seen.decode(errors="replace")


@pytest.mark.skipif(not sys.platform.startswith(("darwin", "linux")), reason="needs a pseudo-terminal")
def test_a_terminal_shows_the_list_and_takes_the_answer(tmp_path):
    out = tmp_path / "flags"
    script = (f'. "{LIB}"; checklist_wanted || {{ echo NO-LIST; exit 3; }}; '
              f'f="$(checklist_run "$(checklist_rows_cuda aarch64)" 800 T "bash oneshot-setup-spark.sh")" && printf %s "$f" > "{out}"')
    code, seen = in_terminal(script)
    assert code == 0 and "NO-LIST" not in seen and "Optional, larger" in seen
    assert out.read_text() == SPARK_OPTIONAL.replace("--music ", "")
    code, seen = in_terminal(f'. "{LIB}"; checklist_wanted --yes && echo LIST || echo NO-LIST')
    assert "NO-LIST" in seen


@pytest.mark.skipif(sys.platform != "darwin", reason="the Mac installer")
def test_the_mac_installer_turns_the_list_into_its_own_flags():
    def dry(*args, answers="\\n"):
        done = subprocess.run([BASH, str(ROOT / "oneshot-setup.sh"), *args], capture_output=True, text=True, timeout=30,
                              env=env(STUDIO_CHECKLIST_DRY="1", STUDIO_CHECKLIST_INPUT=answers))
        assert done.returncode == 0, done.stderr
        return done.stdout.strip(), done.stderr
    flags, drawn = dry("--music")
    assert flags == "--music" and "Optional, larger" not in drawn
    flags, drawn = dry("--app")
    ram = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout) // 2**30
    if ram >= 64:
        assert "Optional, larger" in drawn and flags.startswith("--app --motion --voice --music")
    else:  # below 64 GB the install itself says what this Mac can run
        assert flags == "--app" and "Optional, larger" not in drawn


@pytest.mark.parametrize("script", ["oneshot-setup.sh", "oneshot-setup-cuda.sh", "oneshot-setup-spark.sh",
                                    "oneshot-setup-rtx-6000.sh", "scripts/make-app.sh", "scripts/install-checklist.sh"])
def test_the_scripts_parse(script):
    assert subprocess.run([BASH, "-n", str(ROOT / script)]).returncode == 0


SAO = "https://huggingface.co/stabilityai/stable-audio-open-1.0/resolve/main/model_index.json"
KLEIN = "https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy/resolve/main/diffusion_models/flux3-image-klein-bf16.safetensors"


def gate(tmp_path, *urls, mode="install", **extra):
    """checklist_gate_part for FLUX3-K: its status, what it said, and what the end of the run says."""
    script = (f'. "{LIB}"; checklist_gate_part {mode} --flux3 FLUX3-K "bash oneshot-setup-spark.sh" "$@"; s=$?; '
              'printf "STATUS=%s\\n%s" "$s" "$CHECKLIST_LATER"')
    done = subprocess.run([BASH, "-c", script, "x", *urls], capture_output=True, text=True, timeout=120,
                          env=env(HF_HOME=str(tmp_path), PREFIX=str(tmp_path), **extra))
    return int(re.search(r"STATUS=(\d+)", done.stdout).group(1)), done.stdout


def test_a_gated_part_whose_licence_is_accepted_is_kept(tmp_path):
    code, said = gate(tmp_path, KLEIN, STUDIO_GATE_FAKE="klein=granted")
    assert code == 0 and "Not installed yet" not in said and "!" not in said
    assert gate(tmp_path, KLEIN, STUDIO_GATE_FAKE="klein=open")[0] == 0  # a repo that is not gated


def test_a_refused_licence_drops_the_part_with_what_to_do(tmp_path):
    code, said = gate(tmp_path, KLEIN, STUDIO_GATE_FAKE="klein=accept")
    assert code == 1 and "left out of this install" in said and "Everything else installs" in said
    assert "https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy" in said and "hf auth login" in said
    assert re.search(r"Not installed yet: FLUX3-K \(accept its licence at https://huggingface.co/black-forest-labs/"
                     r"flux-3-image-klein-comfy .*then run: bash oneshot-setup-spark.sh --flux3\)", said)
    assert "sign in to Hugging Face first" in gate(tmp_path, KLEIN, STUDIO_GATE_FAKE="klein=sign_in")[1]
    assert "waiting for the author's approval" in gate(tmp_path, KLEIN, STUDIO_GATE_FAKE="klein=pending")[1]
    code, said = gate(tmp_path, KLEIN, mode="verify", STUDIO_GATE_FAKE="klein=accept")
    assert code == 1 and "FLUX3-K: waiting for the licence" in said and "Not installed yet" not in said


def test_no_network_or_no_sign_in_is_not_yet(tmp_path):
    dead = {"HTTPS_PROXY": "http://127.0.0.1:9", "https_proxy": "http://127.0.0.1:9", "NO_PROXY": "", "no_proxy": ""}
    code, said = gate(tmp_path, SAO, HF_TOKEN="hf_not_a_real_token", **dead)  # app/hf_gate.py itself, with no way out
    assert code == 1 and "Hugging Face did not answer" in said and "Not installed yet: FLUX3-K" in said
    code, said = gate(tmp_path, SAO, **dead)  # no token saved and none given
    assert code == 1 and "sign in to Hugging Face first" in said


def test_the_cuda_installer_asks_before_downloading_and_the_rtx_entry_no_longer_stops():
    cuda = (ROOT / "oneshot-setup-cuda.sh").read_text()
    assert cuda.index("checklist_gate_part \"$GATE_MODE\" --flux3") < cuda.index("say \"Environment:")
    assert "|| FLUX3=0" in cuda and "|| SOUND=0" in cuda and "trap checklist_said_later EXIT" in cuda
    assert "exit 1" not in (ROOT / "oneshot-setup-rtx-6000.sh").read_text()


@pytest.mark.skipif(sys.platform != "darwin", reason="the Mac installer")
def test_the_mac_installer_drops_a_refused_part_and_installs_the_rest(tmp_path):
    done = subprocess.run([BASH, str(ROOT / "oneshot-setup.sh"), "--flux3", "--sound"], capture_output=True, text=True,
                          timeout=60, env=env(STUDIO_GATE_DRY="1", HF_HOME=str(tmp_path), PREFIX=str(tmp_path),
                                              FLUX3_MODEL_DIR=str(tmp_path / "flux3"),
                                              STUDIO_GATE_FAKE="flux-3-image-klein=granted;stable-audio=accept"))
    assert done.returncode == 0, done.stderr
    assert "flux3=1 sound=0" in done.stdout  # FLUX3-K stays, sound waits for its licence
    assert "Sound effects and foley is gated on Hugging Face and left out" in done.stdout
    assert "Not installed yet: Sound effects and foley" in done.stdout and "then run: bash oneshot-setup.sh --sound)" in done.stdout
