# SiLVANN on an Android phone — experimental

SiLVANN runs on a phone's Arm cores, inside [Termux](https://termux.dev): Qwen 3.5 2B `D4` answers as Hugging Face's
own forward of the model does, at **8.0-9.6 tokens a second** on a Razer Edge (Snapdragon G3x Gen 1), a phone-class
handheld from 2023. It is a preview of a preview: one device has run it, the steps are a build from source in a terminal,
and nothing here is an app.

| Phone | Visible RAM | Model | Setup | Prefill | Decode |
|---|---|---|---|---|---|
| Razer Edge — Snapdragon G3x Gen 1, 8 Arm cores, Android 12 | 5.4 GB | [Qwen 3.5 2B](models.md#qwen-35-2b) `D4` | `SILVANN_ARM_INT8=1` | 15.2-15.9 positions/s | 8.02-9.60 tokens/s after 163 |
| the same | 5.4 GB | Qwen 3.5 2B `D4` | exact | 14.7-18.0 positions/s | 4.21-5.36 tokens/s after 163 |

What it uses: the engine and its Arm family, `arm_neon` — Armv8.2 with the dot product, which most phones from 2019 on
have. `SILVANN_ARM_INT8=1` multiplies the weights in integers (`sdot`): about 1.8× the exact route there, the answers as close to
Hugging Face's. The phone's GPU is not used.

**The model for a phone is the 2B.** Its pack is text only and holds about 1.9 GB once loaded (2.5 GB at the load's
peak). The 4B (3.5 GB) did not fit in what Android left free on an 8 GB phone; a phone with 12 GB or more may hold it.

## Setting it up

1. **Termux**, from [F-Droid](https://f-droid.org/packages/com.termux/) (the Play Store's copy is out of date). In it:
   ```
   pkg upgrade
   pkg install clang python python-numpy git rust patchelf
   pip install pybind11
   ```
2. **The Python packages Termux does not carry.** `tokenizers` and FastAPI's `pydantic-core` have no Android builds:
   pip builds them from source with Rust (about ten minutes each), and the libraries it makes must then name Python's,
   which Android's loader needs and pip's build leaves out:
   ```
   export ANDROID_API_LEVEL=24
   pip install tokenizers fastapi uvicorn httpx huggingface_hub python-multipart
   SITE=$(python -c 'import site; print(site.getsitepackages()[0])')
   LIBPY=libpython$(python -c 'import sysconfig; print(sysconfig.get_config_var("LDVERSION"))').so
   for so in $SITE/tokenizers/*.so $SITE/pydantic_core/*.so; do patchelf --add-needed $LIBPY $so; done
   ```
3. **SiLVANN, built for the phone's cores:**
   ```
   git clone https://github.com/eval-apply-labs/SiLVANN && cd SiLVANN
   SILVANN_SILICON_FAMILIES=arm_neon bash scripts/build.sh
   ```
   It ends with `== built: the evaluator, and the families arm_neon (loaded)`.
4. **The model** (1.7 GB), over plain HTTPS — Hugging Face's faster Xet client cannot check a server's certificate outside
   an Android app, so it is switched off for this download:
   ```
   HF_HUB_DISABLE_XET=1 python silvann_core.py download-model silvann_tq_D4_qwen3.5_2b
   ```
5. **The server, then the chat** — the server in one Termux session, the chat in a second (swipe from the left edge, *New
   session*):
   ```
   SILVANN_ARM_INT8=1 python start_silvann_server.py --model models/silvann_tq_D4_qwen3.5_2b
   python silvann_cli.py
   ```

## Keeping Android from stopping it

- **Android stops Termux's work in the background**, and from Android 12 the processes it starts past a small number.
  Pull down Termux's notification and tap *Acquire wakelock*; set Termux's battery use to *Unrestricted*. On Android 12L
  and later, *Developer options → Disable child process restrictions*; on Android 12 itself, from a computer with `adb`:
  `adb shell "/system/bin/device_config set_sync_disabled_for_tests persistent; /system/bin/device_config put activity_manager max_phantom_processes 2147483647"`.
- **Memory:** close the other apps before the model loads. When the phone runs short, Android closes Termux whole —
  every session in it.
- **Heat:** a phone slows its cores within minutes of full use; on the Razer Edge the same model ran 5.5 tokens a second
  cool and 4.8 hot through the exact route.

## Not there yet

The UI (`silvann_ui.py`) on the phone, an app, the phone's GPU, and the Arm family's own code for attention, DeltaNet
and the norms, which still run nn's generic body there. The measurements, their commands and the device:
`measurements/2026-11-10_nn171_razer_edge.md` in the development repository.
